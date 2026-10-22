// HydroGym <-> NekRS control interface (udf side), in the spirit of hydrogym/maia/mpmd_interface.py:
// probe locations, actuator layout (positions, footprint, blowing direction), zero-net-mass-flux and the
// quantities monitored for the reward are all defined in Python (HydroGym YAML) and sent at run time.
// The case only marks the boundaries that may be actuated:
//
//   [CASEDATA]
//   hydrogymBoundaries = 1, 3     # boundary ids with velocity boundaryTypeMap = udfDirichlet
//
// and includes this header from its udf (see examples/nekrs/channel/tcf.udf):
//
//   #include "../../../hydrogym/nekrs/udf/hydrogym_nekrs.hpp"   // path relative to the case directory
//   #ifdef __okl__
//   void udfDirichlet(bcData *bc)
//   {
//     if (isField("fluid velocity")) {
//       bc->uxFluid = bc->usrwrk[bc->idxVol + 0 * bc->fieldOffset];
//       bc->uyFluid = bc->usrwrk[bc->idxVol + 1 * bc->fieldOffset];
//       bc->uzFluid = bc->usrwrk[bc->idxVol + 2 * bc->fieldOffset];
//     }
//   }
//   #endif
//   void UDF_Setup0(MPI_Comm comm, setupAide &options) { hgnek::setup0(); }
//   void UDF_Setup() { hgnek::setup(); }
//   void UDF_ExecuteStep(double time, int tstep) { hgnek::executeStep(time, tstep); }
//
// Wire protocol (intercommunicator with the controller, created with tag 99 as in hydrogym/nek/env.py).
// Every request is   command (5 x MPI_CHAR, tag 22), int payload (MPI_INT, tag 23), double payload (tag 24)
// Every reply is     ints [status, ...] (tag 25), doubles (tag 26), message (MPI_CHAR, tag 27, may be empty)
// status 0 = ok (a non-empty message is a warning), otherwise the request was rejected and nothing changed.
//
//   HELLO  -> ints [0, version, nranks, N, nAllowed, allowed bids...], dbl [time, dt, mu, rho, tstep]
//   BNDRY  ints [bid, mode] (mode 0: all GLL face nodes incl. duplicates, legacy order; 1: legacy facindf;
//          2: mode 0 with duplicate positions removed)
//          -> ints [0, n, (rank, element+1, Nek face, ix, iy, iz) x n], dbl [(x, y, z, nx, ny, nz) x n]
//   PRBDF  ints [n, nf, field codes (0 u, 1 v, 2 w, 3 p) x nf, meanAxis (-1 off, 0/1/2 = subtract the mean
//          over planes normal to x/y/z)], dbl [(x, y, z) x n]  -> ints [0, nBorder], dbl [max distance]
//   PROBE  -> ints [0], dbl [time, CFL, values (field-major, nf x n)]
//   ACTDF  ints [n, nb, bids x nb, znmf (0 none, 1 uniform over the bids, 2 remove the flux-weighted mean
//          action), check (0 off, 1 warn, 2 error), footprint x n (0 point, 1 gaussian, 2 box),
//          direction x n (0 wall normal, 1 vector, 2 angle from the normal towards a vector)]
//          dbl [minNodesPerWidth, pointTol, centre x 3n, width x 3n (sigma or half width, 0 = unbounded),
//          vector x 3n, angle [rad] x n]
//          -> ints [0, nodes x n, flags x n (1 = under-resolved)], dbl [flux per unit action x n,
//          nodes per width x n, area]
//   ACTUA dbl [velocity amplitude x n] -> dbl [net flux, max|u_wall|, znmf correction, max|a|]
//   MONDF  ints [nm, (bid, kind) x nm] (kind 0: viscous + pressure force, 1: mean d(u,v,w)/dn) -> dbl [area x nm]
//   STEPS  ints [n] -> after n solver steps: ints [0, n, row length], dbl rows
//          [tstep, time, dt, CFL, 6 values per monitor]
//   SNAPS  ints [slot] -> ints [0, slots in use], dbl [time, bytes]: copy the complete time-stepper state
//          (all lags of u, p, explicit and advection terms, dt[3]) to device memory; needs >= 3 steps of history
//   RESET  ints [0, slot] restore a snapshot, or [1, chars of "file[+int][+time=t]"] load a field file
//          (u, p; the lag history is rebuilt from the loaded state) -> dbl [solver time, state time].
//          The solver clock keeps running, the actuation is set to zero.
//   SNAPF  ints [op (0 write, 1 read), slot, chars of path] -> ints [0, tstep], dbl [time]: snapshot slot to or
//          from path_r<rank> (binary, checked against partition, order and mesh on read)
//   TERMN  stop (no reply)

#ifndef __okl__

#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include "pointInterpolation.hpp"
#include "opSEM.hpp"
#include "mesh3D.h"

namespace hgnek
{

static constexpr int version = 2;
static constexpr int tagCmd = 22, tagInt = 23, tagDbl = 24, tagRInt = 25, tagRDbl = 26, tagRMsg = 27;

static MPI_Comm inter = MPI_COMM_NULL;
static std::vector<int> allowed;
static double curTime = 0;
static int curStep = 0;

// boundary GLL nodes of all faces (one entry per element face copy)
struct bNode {
  dlong vol;     // volume index
  int bid;
  double x[3];
  double n[3];   // unit normal pointing into the fluid
  double w;      // surface quadrature weight (WsJ)
  double h[3];   // GLL spacing along x, y, z (from the face direction best aligned with it)
  double hmin;   // smaller of the two face spacings
};
static std::vector<bNode> bnodes;
static std::vector<std::vector<int>> faceBNode; // [e * Nfaces + f] -> index of face node 0 in bnodes, or -1

static double hminAll = 0;                       // smallest boundary GLL spacing
static std::vector<dfloat> vgeo, idH;           // for the CFL

// probes
static std::unique_ptr<pointInterpolation_t> interp;
static int nProbe = 0;
static std::vector<int> probeFields;
static int meanAxis = -1;
static std::vector<int> nodeLevel;
static std::vector<double> levelWeight;
static int nLevels = 0;

// actuators
struct actEntry {
  int node; // index in bnodes
  int act;
  double w[3]; // footprint * direction
};
static std::vector<actEntry> entries;
static int nAct = 0;
static std::vector<int> actBids;
static int znmf = 0;
static std::vector<double> actFlux; // flux per unit action
static std::vector<int> actNodes;   // indices in bnodes on the actuated boundaries
static double actArea = 0;

// monitors and stepping
static std::vector<std::pair<int, int>> monitors;
static std::vector<occa::memory> o_monBid;
static std::vector<double> monArea;
static int stepsLeft = 0;
static std::vector<double> rows;

static MPI_Comm comm()
{
  return platform->comm.mpiComm();
}
static int rank()
{
  return platform->comm.mpiRank();
}

// ---------------------------------------------------------------- messaging

template <typename T> static std::vector<T> recvPayload(int tag, MPI_Datatype type)
{
  int n = 0;
  std::vector<T> v;
  if (rank() == 0) {
    MPI_Status st;
    MPI_Probe(0, tag, inter, &st);
    MPI_Get_count(&st, type, &n);
    v.resize(n);
    MPI_Recv(v.data(), n, type, 0, tag, inter, MPI_STATUS_IGNORE);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, comm());
  v.resize(n);
  if (n) MPI_Bcast(v.data(), n, type, 0, comm());
  return v;
}

static void reply(int status, std::vector<int> ints, const std::vector<double> &dbls, const std::string &msg = "")
{
  if (rank() != 0) return;
  ints.insert(ints.begin(), status);
  MPI_Send(ints.data(), ints.size(), MPI_INT, 0, tagRInt, inter);
  MPI_Send(dbls.data(), dbls.size(), MPI_DOUBLE, 0, tagRDbl, inter);
  MPI_Send(msg.data(), msg.size(), MPI_CHAR, 0, tagRMsg, inter);
  if (!msg.empty()) std::cout << "[HGNEK] " << (status ? "ERROR: " : "WARNING: ") << msg << std::endl;
}

// gather variable-length local arrays on rank 0 in rank order
template <typename T> static std::vector<T> gatherv(const std::vector<T> &loc, MPI_Datatype type)
{
  const int size = platform->comm.mpiCommSize();
  int n = loc.size();
  std::vector<int> counts(size), displ(size, 0);
  MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm());
  std::vector<T> all;
  if (rank() == 0) {
    for (int r = 1; r < size; r++) displ[r] = displ[r - 1] + counts[r - 1];
    all.resize(displ[size - 1] + counts[size - 1]);
  }
  MPI_Gatherv(loc.data(), n, type, all.data(), counts.data(), displ.data(), type, 0, comm());
  return all;
}

// ---------------------------------------------------------------- geometry

static void setupGeometry()
{
  auto mesh = nrs->meshV;
  const int Nq = mesh->Nq, Nfp = mesh->Nfp, Nf = mesh->Nfaces;
  const auto [x, y, z] = mesh->xyzHost();
  const std::vector<dfloat> *c[3] = {&x, &y, &z};

  const dlong nFaceNodes = mesh->Nelements * Nf * Nfp;
  std::vector<dfloat> sgeo(nFaceNodes * Nsgeo);
  mesh->o_sgeo.copyTo(sgeo.data(), sgeo.size());
  std::vector<dlong> vmapM(nFaceNodes);
  mesh->o_vmapM.copyTo(vmapM.data(), vmapM.size());

  faceBNode.assign(mesh->Nelements * Nf, {});
  for (dlong e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < Nf; f++) {
      const int bid = mesh->EToB[f + e * Nf];
      if (bid <= 0) continue;
      faceBNode[e * Nf + f] = {static_cast<int>(bnodes.size())};
      for (int t = 0; t < Nfp; t++) {
        const dlong sid = e * Nf * Nfp + f * Nfp + t;
        bNode b;
        b.vol = vmapM[sid];
        b.bid = bid;
        for (int d = 0; d < 3; d++) {
          b.x[d] = (*c[d])[b.vol];
          b.n[d] = -sgeo[sid * Nsgeo + NXID + d];
        }
        b.w = sgeo[sid * Nsgeo + WSJID];

        // spacing vectors along the two face directions (centred inside, one-sided at the edges)
        const int ia = t % Nq, ib = t / Nq;
        double ev[2][3];
        for (int dir = 0; dir < 2; dir++) {
          const int i = dir ? ib : ia;
          const int stride = dir ? Nq : 1;
          const int im = std::max(i - 1, 0), ip = std::min(i + 1, Nq - 1);
          const dlong vm = vmapM[sid + (im - i) * stride], vp = vmapM[sid + (ip - i) * stride];
          for (int d = 0; d < 3; d++) ev[dir][d] = ((*c[d])[vp] - (*c[d])[vm]) / (ip - im);
        }
        double len[2];
        for (int dir = 0; dir < 2; dir++) len[dir] = std::sqrt(ev[dir][0] * ev[dir][0] + ev[dir][1] * ev[dir][1] + ev[dir][2] * ev[dir][2]);
        b.hmin = std::min(len[0], len[1]);
        for (int d = 0; d < 3; d++) {
          double h = 1e300;
          for (int dir = 0; dir < 2; dir++) {
            const double cosd = std::abs(ev[dir][d]) / len[dir];
            if (cosd > 1e-3) h = std::min(h, len[dir] / cosd);
          }
          b.h[d] = h;
        }
        bnodes.push_back(b);
      }
    }
  }

  hminAll = 1e300;
  for (const auto &b : bnodes) hminAll = std::min(hminAll, b.hmin);
  MPI_Allreduce(MPI_IN_PLACE, &hminAll, 1, MPI_DOUBLE, MPI_MIN, comm());

  vgeo.resize(static_cast<size_t>(mesh->Nelements) * mesh->Np * Nvgeo);
  mesh->o_vgeo.copyTo(vgeo.data(), vgeo.size());
  idH.resize(Nq);
  for (int n = 0; n < Nq; n++) {
    const double dx = (n == 0)        ? mesh->gllz[1] - mesh->gllz[0]
                      : (n == Nq - 1) ? mesh->gllz[n] - mesh->gllz[n - 1]
                                      : 0.5 * (mesh->gllz[n + 1] - mesh->gllz[n - 1]);
    idH[n] = 1.0 / dx;
  }
}

// same definition as nrs_t::computeCFL (cflHex3D), evaluated on the host (computeCFL is private)
static double computeCFL()
{
  auto mesh = nrs->meshV;
  const int Nq = mesh->Nq, Np = mesh->Np;
  const auto offset = nrs->fluid->fieldOffset;
  std::vector<dfloat> U(3 * offset);
  nrs->fluid->o_U.copyTo(U.data(), 3 * offset);
  const double dt = nrs->dt[0];
  double cfl = 0;
  for (dlong e = 0; e < mesh->Nelements; e++) {
    for (int n = 0; n < Np; n++) {
      const dlong id = e * Np + n;
      const dlong g = e * Np * Nvgeo + n;
      const int i = n % Nq, j = (n / Nq) % Nq, k = n / (Nq * Nq);
      const double u = U[id], v = U[id + offset], w = U[id + 2 * offset];
      const double ur = std::abs(u * vgeo[g + RXID * Np] + v * vgeo[g + RYID * Np] + w * vgeo[g + RZID * Np]) * idH[i];
      const double us = std::abs(u * vgeo[g + SXID * Np] + v * vgeo[g + SYID * Np] + w * vgeo[g + SZID * Np]) * idH[j];
      const double ut = std::abs(u * vgeo[g + TXID * Np] + v * vgeo[g + TYID * Np] + w * vgeo[g + TZID * Np]) * idH[k];
      cfl = std::max(cfl, (ur + us + ut) * dt);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &cfl, 1, MPI_DOUBLE, MPI_MAX, comm());
  return cfl;
}

// ---------------------------------------------------------------- requests

static void cmdHello()
{
  auto mesh = nrs->meshV;
  dfloat mue = 0, rho = 0;
  platform->options.getArgs("FLUID VISCOSITY", mue);
  platform->options.getArgs("FLUID DENSITY", rho);
  std::vector<int> ints = {version, platform->comm.mpiCommSize(), mesh->N, static_cast<int>(allowed.size())};
  ints.insert(ints.end(), allowed.begin(), allowed.end());
  reply(0, ints, {curTime, static_cast<double>(nrs->dt[0]), mue, rho, static_cast<double>(curStep)});
}

// boundary GLL nodes in the order of the Nek5000 HydroGym registration (register_wall_pts):
// local elements, Nek5000 faces 1..6, loops iz, iy, ix over facind (mode 0/2) or facindf (mode 1)
static void cmdBndry(const std::vector<int> &in)
{
  auto mesh = nrs->meshV;
  const int N = mesh->N, Nq = mesh->Nq;
  if (in.size() != 2) return reply(1, {}, {}, "BNDRY expects [bid, mode]");
  const int bid = in[0], mode = in[1];

  const int nekFace[6] = {5, 1, 2, 3, 4, 6}; // NekRS face -> Nek5000 face
  std::vector<int> li;
  std::vector<double> ld;
  for (dlong e = 0; e < mesh->Nelements; e++) {
    const auto eg = nek::localElementIdToGlobal(e);
    for (int nf = 1; nf <= 6; nf++) {
      int f = 0;
      while (nekFace[f] != nf) f++;
      if (mesh->EToB[f + e * mesh->Nfaces] != bid) continue;
      int lo[3] = {0, 0, 0};
      int hi[3] = {N, N, N};
      if (mode == 1) hi[0] = hi[1] = hi[2] = N - 1;
      const int dir = (nf == 2 || nf == 4) ? 0 : (nf == 1 || nf == 3) ? 1 : 2;
      lo[dir] = hi[dir] = (nf == 1 || nf == 4 || nf == 5) ? 0 : N;

      // volume index -> face node
      std::vector<int> v2t(mesh->Np, -1);
      for (int t = 0; t < mesh->Nfp; t++) v2t[mesh->faceNodes[f * mesh->Nfp + t]] = t;
      const int b0 = faceBNode[e * mesh->Nfaces + f][0];

      for (int k = lo[2]; k <= hi[2]; k++) {
        for (int j = lo[1]; j <= hi[1]; j++) {
          for (int i = lo[0]; i <= hi[0]; i++) {
            const auto &b = bnodes[b0 + v2t[k * Nq * Nq + j * Nq + i]];
            li.insert(li.end(), {rank(), static_cast<int>(eg + 1), nf, i + 1, j + 1, k + 1});
            ld.insert(ld.end(), {b.x[0], b.x[1], b.x[2], b.n[0], b.n[1], b.n[2]});
          }
        }
      }
    }
  }

  auto ai = gatherv(li, MPI_INT);
  auto ad = gatherv(ld, MPI_DOUBLE);
  if (rank() != 0) return;

  int n = ad.size() / 6;
  if (mode == 2) {
    // drop repeated positions (shared edges and corners), keep the first occurrence
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    auto key = [&](int p) { return std::array<double, 3>{ad[6 * p], ad[6 * p + 1], ad[6 * p + 2]}; };
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key(a) < key(b); });
    std::vector<char> keep(n, 1);
    const double tol = 1e-6 * hminAll;
    for (int q = 1; q < n; q++) {
      for (int r = q - 1; r >= 0; r--) {
        const int a = idx[r], b = idx[q];
        if (ad[6 * b] - ad[6 * a] > tol) break;
        if (keep[a] && std::abs(ad[6 * b + 1] - ad[6 * a + 1]) <= tol && std::abs(ad[6 * b + 2] - ad[6 * a + 2]) <= tol) {
          keep[std::max(a, b)] = 0;
        }
      }
    }
    std::vector<int> ri;
    std::vector<double> rd;
    for (int p = 0; p < n; p++) {
      if (!keep[p]) continue;
      ri.insert(ri.end(), ai.begin() + 6 * p, ai.begin() + 6 * p + 6);
      rd.insert(rd.end(), ad.begin() + 6 * p, ad.begin() + 6 * p + 6);
    }
    ai.swap(ri);
    ad.swap(rd);
    n = ad.size() / 6;
  }
  if (n == 0) return reply(1, {}, {}, "BNDRY: boundary id " + std::to_string(bid) + " has no faces");
  ai.insert(ai.begin(), n);
  reply(0, ai, ad);
}

// GLL planes normal to the axis (needed to subtract the homogeneous-plane mean, as planar_avg in Nek5000)
static std::string setupLevels(int axis)
{
  auto mesh = nrs->meshV;
  const auto [x, y, z] = mesh->xyzHost();
  const auto &c = (axis == 0) ? x : (axis == 1) ? y : z;
  const double tol = 1e-9;

  int bad = 0;
  std::vector<double> loc;
  for (dlong e = 0; e < mesh->Nelements; e++) {
    std::vector<double> v(c.begin() + e * mesh->Np, c.begin() + (e + 1) * mesh->Np);
    std::sort(v.begin(), v.end());
    std::vector<double> u;
    for (auto a : v) {
      if (u.empty() || a - u.back() > tol) u.push_back(a);
    }
    if ((int)u.size() != mesh->Nq) bad++;
    loc.insert(loc.end(), u.begin(), u.end());
  }
  MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, comm());
  if (bad) {
    return std::to_string(bad) + " elements do not have their GLL nodes on planes normal to axis " +
           std::to_string(axis) + "; plane means need an extruded mesh";
  }

  const int size = platform->comm.mpiCommSize();
  int nl = loc.size();
  std::vector<int> counts(size), displ(size, 0);
  MPI_Allgather(&nl, 1, MPI_INT, counts.data(), 1, MPI_INT, comm());
  for (int r = 1; r < size; r++) displ[r] = displ[r - 1] + counts[r - 1];
  std::vector<double> all(displ[size - 1] + counts[size - 1]);
  MPI_Allgatherv(loc.data(), nl, MPI_DOUBLE, all.data(), counts.data(), displ.data(), MPI_DOUBLE, comm());
  std::sort(all.begin(), all.end());
  std::vector<double> levels;
  for (auto a : all) {
    if (levels.empty() || a - levels.back() > tol) levels.push_back(a);
  }
  nLevels = levels.size();

  std::vector<dfloat> w(mesh->Nlocal);
  mesh->o_LMM.copyTo(w.data(), mesh->Nlocal);
  nodeLevel.resize(mesh->Nlocal);
  levelWeight.assign(nLevels, 0.0);
  for (dlong n = 0; n < mesh->Nlocal; n++) {
    nodeLevel[n] = std::lower_bound(levels.begin(), levels.end(), c[n] - tol) - levels.begin();
    levelWeight[nodeLevel[n]] += w[n];
  }
  MPI_Allreduce(MPI_IN_PLACE, levelWeight.data(), nLevels, MPI_DOUBLE, MPI_SUM, comm());
  return "";
}

static void cmdPrbdf(const std::vector<int> &in, const std::vector<double> &din)
{
  auto mesh = nrs->meshV;
  if (in.size() < 3 || (int)in.size() != 3 + in[1]) return reply(1, {}, {}, "PRBDF: malformed int payload");
  const int n = in[0], nf = in[1];
  if ((int)din.size() != 3 * n) return reply(1, {}, {}, "PRBDF: expected 3n coordinates");
  std::vector<int> fields(in.begin() + 2, in.begin() + 2 + nf);
  for (auto f : fields) {
    if (f < 0 || f > 3) return reply(1, {}, {}, "PRBDF: field codes are 0 u, 1 v, 2 w, 3 p");
  }
  const int axis = in[2 + nf];
  if (axis >= 0) {
    const auto err = setupLevels(axis);
    if (!err.empty()) return reply(1, {}, {}, "PRBDF: " + err);
  }

  std::vector<dfloat> xs, ys, zs;
  if (rank() == 0) {
    for (int p = 0; p < n; p++) {
      xs.push_back(din[3 * p]);
      ys.push_back(din[3 * p + 1]);
      zs.push_back(din[3 * p + 2]);
    }
  }
  auto ip = std::make_unique<pointInterpolation_t>(mesh, comm());
  ip->setPoints(xs, ys, zs);
  ip->find(pointInterpolation_t::VerbosityLevel::None);

  int nBorder = 0, nLost = 0;
  double maxDist = 0;
  std::string lost;
  if (rank() == 0) {
    const auto &d = ip->data();
    for (int p = 0; p < n; p++) {
      if (d.code[p] == pointInterpolation_t::CODE_BORDER) nBorder++;
      if (d.code[p] == pointInterpolation_t::CODE_NOT_FOUND) {
        if (nLost++ < 5) lost += " " + std::to_string(p);
      }
      maxDist = std::max(maxDist, std::sqrt(static_cast<double>(d.dist2[p])));
    }
  }
  MPI_Bcast(&nLost, 1, MPI_INT, 0, comm());
  if (nLost) {
    return reply(1, {}, {}, "PRBDF: " + std::to_string(nLost) + " probes outside the mesh, e.g. index" + lost);
  }

  interp = std::move(ip);
  nProbe = n;
  probeFields = fields;
  meanAxis = axis;
  if (rank() == 0) {
    std::cout << "[HGNEK] PRBDF: " << n << " probes, " << nf << " fields, meanAxis " << axis << ", " << nBorder
              << " on element borders" << std::endl;
  }
  reply(0, {nBorder}, {maxDist});
}

static void cmdProbe()
{
  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  const double cfl = computeCFL();
  if (!interp) return reply(1, {}, {}, "PROBE before PRBDF");

  const int nf = probeFields.size();
  std::vector<dfloat> w;
  if (meanAxis >= 0) {
    w.resize(mesh->Nlocal);
    mesh->o_LMM.copyTo(w.data(), mesh->Nlocal);
  }
  std::vector<dfloat> fld(nf * offset, 0.0);
  for (int c = 0; c < nf; c++) {
    std::vector<dfloat> u(mesh->Nlocal);
    if (probeFields[c] < 3) {
      nrs->fluid->o_U.copyTo(u.data(), mesh->Nlocal, probeFields[c] * offset);
    } else {
      nrs->fluid->o_P.copyTo(u.data(), mesh->Nlocal);
    }
    if (meanAxis >= 0) {
      std::vector<double> m(nLevels, 0.0);
      for (dlong n = 0; n < mesh->Nlocal; n++) m[nodeLevel[n]] += w[n] * u[n];
      MPI_Allreduce(MPI_IN_PLACE, m.data(), nLevels, MPI_DOUBLE, MPI_SUM, comm());
      for (int l = 0; l < nLevels; l++) m[l] /= levelWeight[l];
      for (dlong n = 0; n < mesh->Nlocal; n++) u[n] -= m[nodeLevel[n]];
    }
    std::copy(u.begin(), u.end(), fld.begin() + c * offset);
  }

  auto o_fld = platform->deviceMemoryPool.reserve<dfloat>(nf * offset);
  o_fld.copyFrom(fld.data(), nf * offset);
  const int nLoc = (rank() == 0) ? nProbe : 0;
  const int outOff = std::max(nLoc, 1);
  auto o_out = platform->deviceMemoryPool.reserve<dfloat>(nf * outOff);
  interp->eval(nf, offset, o_fld, outOff, o_out);

  if (rank() != 0) return;
  std::vector<dfloat> out(nf * outOff);
  o_out.copyTo(out.data(), nf * outOff);
  std::vector<double> d = {curTime, cfl};
  for (int c = 0; c < nf; c++) {
    for (int p = 0; p < nProbe; p++) d.push_back(out[p + c * outOff]);
  }
  reply(0, {}, d);
}

static void cmdActdf(const std::vector<int> &in, const std::vector<double> &din)
{
  if (in.size() < 2) return reply(1, {}, {}, "ACTDF: malformed int payload");
  const int n = in[0], nb = in[1];
  if ((int)in.size() != 2 + nb + 2 + 2 * n || (int)din.size() != 2 + 10 * n) {
    return reply(1, {}, {}, "ACTDF: payload sizes do not match the number of actuators");
  }
  std::vector<int> bids(in.begin() + 2, in.begin() + 2 + nb);
  const int zm = in[2 + nb], check = in[3 + nb];
  const int *ftype = in.data() + 4 + nb;
  const int *dmode = ftype + n;
  const double minNodes = din[0], ptTol = din[1];
  const double *cen = din.data() + 2, *wid = cen + 3 * n, *vec = wid + 3 * n, *ang = vec + 3 * n;

  for (auto b : bids) {
    if (std::find(allowed.begin(), allowed.end(), b) == allowed.end()) {
      return reply(1, {}, {}, "ACTDF: boundary " + std::to_string(b) +
                              " is not listed in [CASEDATA] hydrogymBoundaries of the par file");
    }
  }

  std::vector<int> nodes;
  for (int k = 0; k < (int)bnodes.size(); k++) {
    if (std::find(bids.begin(), bids.end(), bnodes[k].bid) != bids.end()) nodes.push_back(k);
  }

  std::vector<actEntry> ent;
  std::vector<double> flux(n, 0.0), nbar(3 * n, 0.0), hmax(3 * n, 0.0), hmaxAll(3 * n, 0.0);
  std::vector<int> count(n, 0), core(n, 0);
  std::string dirErr;
  for (int i = 0; i < n; i++) {
    double reach[3];
    for (int d = 0; d < 3; d++) {
      const double s = wid[3 * i + d];
      reach[d] = (ftype[i] == 0 || s <= 0) ? 1e300 : (ftype[i] == 1) ? 5.3 * s : s * (1 + 1e-12);
    }
    for (int k : nodes) {
      const auto &b = bnodes[k];
      double phi = 0;
      if (ftype[i] == 0) {
        double r2 = 0;
        for (int d = 0; d < 3; d++) r2 += (b.x[d] - cen[3 * i + d]) * (b.x[d] - cen[3 * i + d]);
        phi = (std::sqrt(r2) <= ptTol * b.hmin) ? 1 : 0;
      } else {
        bool inside = true;
        double q = 0;
        for (int d = 0; d < 3 && inside; d++) {
          const double dx = b.x[d] - cen[3 * i + d];
          if (std::abs(dx) > reach[d]) inside = false;
          if (ftype[i] == 1 && wid[3 * i + d] > 0) q += dx * dx / (wid[3 * i + d] * wid[3 * i + d]);
        }
        if (!inside) continue;
        phi = (ftype[i] == 1) ? std::exp(-0.5 * q) : 1;
      }
      if (phi <= 0) continue;

      double dir[3];
      if (dmode[i] == 0) {
        for (int d = 0; d < 3; d++) dir[d] = b.n[d];
      } else {
        double v[3], vn = 0, len = 0;
        for (int d = 0; d < 3; d++) vn += vec[3 * i + d] * b.n[d];
        for (int d = 0; d < 3; d++) {
          v[d] = (dmode[i] == 1) ? vec[3 * i + d] : vec[3 * i + d] - vn * b.n[d];
          len += v[d] * v[d];
        }
        len = std::sqrt(len);
        if (len < 1e-8) {
          if (dirErr.empty()) dirErr = "actuator " + std::to_string(i) + ": direction vector is zero or parallel to the wall normal";
          continue;
        }
        for (int d = 0; d < 3; d++) {
          dir[d] = (dmode[i] == 1) ? v[d] / len : std::cos(ang[i]) * b.n[d] + std::sin(ang[i]) * v[d] / len;
        }
      }

      actEntry a;
      a.node = k;
      a.act = i;
      double dn = 0;
      for (int d = 0; d < 3; d++) {
        a.w[d] = phi * dir[d];
        dn += a.w[d] * b.n[d];
        nbar[3 * i + d] += b.w * phi * b.n[d];
      }
      flux[i] += b.w * dn;
      count[i]++;
      for (int d = 0; d < 3; d++) hmaxAll[3 * i + d] = std::max(hmaxAll[3 * i + d], b.h[d]);
      if (phi >= 0.5) {
        core[i]++;
        for (int d = 0; d < 3; d++) hmax[3 * i + d] = std::max(hmax[3 * i + d], b.h[d]);
      }
      ent.push_back(a);
    }
  }

  int dirBad = !dirErr.empty();
  MPI_Allreduce(MPI_IN_PLACE, &dirBad, 1, MPI_INT, MPI_MAX, comm());
  MPI_Allreduce(MPI_IN_PLACE, flux.data(), n, MPI_DOUBLE, MPI_SUM, comm());
  MPI_Allreduce(MPI_IN_PLACE, nbar.data(), 3 * n, MPI_DOUBLE, MPI_SUM, comm());
  MPI_Allreduce(MPI_IN_PLACE, hmax.data(), 3 * n, MPI_DOUBLE, MPI_MAX, comm());
  MPI_Allreduce(MPI_IN_PLACE, hmaxAll.data(), 3 * n, MPI_DOUBLE, MPI_MAX, comm());
  MPI_Allreduce(MPI_IN_PLACE, count.data(), n, MPI_INT, MPI_SUM, comm());
  MPI_Allreduce(MPI_IN_PLACE, core.data(), n, MPI_INT, MPI_SUM, comm());
  if (dirBad) {
    return reply(1, {}, {}, "ACTDF: " + (dirErr.empty() ? std::string("invalid direction vector") : dirErr));
  }

  double area = 0;
  for (int k : nodes) area += bnodes[k].w;
  MPI_Allreduce(MPI_IN_PLACE, &area, 1, MPI_DOUBLE, MPI_SUM, comm());

  // placement and resolution: nodes per footprint width along each tangential direction
  std::vector<int> flags(n, 0);
  std::vector<double> res(n, 0.0);
  std::ostringstream miss, coarse;
  int nMiss = 0, nCoarse = 0;
  for (int i = 0; i < n; i++) {
    if (count[i] == 0) {
      if (nMiss++ < 5) {
        miss << " " << i << " (" << cen[3 * i] << ", " << cen[3 * i + 1] << ", " << cen[3 * i + 2] << ")";
      }
      continue;
    }
    if (ftype[i] == 0) {
      res[i] = -1; // a point actuator is one node by construction
      continue;
    }
    double len = 0;
    for (int d = 0; d < 3; d++) len += nbar[3 * i + d] * nbar[3 * i + d];
    len = std::sqrt(len);
    double r = 1e300;
    for (int d = 0; d < 3; d++) {
      if (wid[3 * i + d] <= 0 || std::abs(nbar[3 * i + d]) > 0.9 * len) continue; // unbounded or wall-normal
      const double width = (ftype[i] == 1) ? 2.3548 * wid[3 * i + d] : 2 * wid[3 * i + d]; // FWHM or full width
      // node spacing in the core (phi >= 0.5); a footprint that falls between nodes uses the nodes it touches
      r = std::min(r, width / (core[i] ? hmax : hmaxAll)[3 * i + d]);
    }
    res[i] = (r == 1e300) ? -1 : r;
    if (core[i] == 0 || (res[i] >= 0 && res[i] < minNodes)) {
      flags[i] = 1;
      if (nCoarse++ < 5) coarse << " " << i << " (" << res[i] << ")";
    }
  }

  if (nMiss) {
    return reply(1, {}, {}, "ACTDF: " + std::to_string(nMiss) +
                            " actuators do not cover any node of the actuated boundaries, e.g." + miss.str());
  }
  std::string warn;
  if (nCoarse && check > 0) {
    std::ostringstream s;
    s << "ACTDF: " << nCoarse << " actuators are under-resolved (< " << minNodes
      << " GLL nodes per footprint width, or no node with phi >= 0.5), e.g. actuator (nodes per width):"
      << coarse.str();
    if (check == 2) return reply(2, {}, {}, s.str() + "; refine the mesh, widen the footprint or set resolution_check to warn");
    warn = s.str();
  }

  double fsum = 0;
  for (auto f : flux) fsum += f;
  if (zm == 2 && std::abs(fsum) < 1e-14 * area) return reply(1, {}, {}, "ACTDF: znmf 2 needs a nonzero total actuator flux");

  entries.swap(ent);
  nAct = n;
  actBids = bids;
  znmf = zm;
  actFlux = flux;
  actNodes = nodes;
  actArea = area;
  platform->linAlg->fill(3 * nrs->fluid->fieldOffset, 0.0, platform->app->bc->o_usrwrk);

  if (rank() == 0) {
    std::cout << "[HGNEK] ACTDF: " << n << " actuators on " << nb << " boundaries, area " << area << ", znmf " << zm
              << std::endl;
  }
  std::vector<int> ri = count;
  ri.insert(ri.end(), flags.begin(), flags.end());
  std::vector<double> rd = flux;
  rd.insert(rd.end(), res.begin(), res.end());
  rd.push_back(area);
  reply(0, ri, rd, warn);
}

static void cmdActua(const std::vector<double> &din)
{
  if (nAct == 0) return reply(1, {}, {}, "ACTUA before ACTDF");
  if ((int)din.size() != nAct) {
    return reply(1, {}, {}, "ACTUA: got " + std::to_string(din.size()) + " values for " + std::to_string(nAct) + " actuators");
  }
  std::vector<double> a = din;
  double shift = 0;
  if (znmf == 2) {
    double fa = 0, fs = 0;
    for (int i = 0; i < nAct; i++) {
      fa += a[i] * actFlux[i];
      fs += actFlux[i];
    }
    shift = fa / fs;
    for (auto &v : a) v -= shift;
  }

  std::vector<double> u(3 * bnodes.size(), 0.0);
  for (const auto &e : entries) {
    for (int d = 0; d < 3; d++) u[3 * e.node + d] += a[e.act] * e.w[d];
  }
  if (znmf == 1) {
    double q = 0;
    for (int k : actNodes) {
      for (int d = 0; d < 3; d++) q += bnodes[k].w * u[3 * k + d] * bnodes[k].n[d];
    }
    MPI_Allreduce(MPI_IN_PLACE, &q, 1, MPI_DOUBLE, MPI_SUM, comm());
    shift = q / actArea;
    for (int k : actNodes) {
      for (int d = 0; d < 3; d++) u[3 * k + d] -= shift * bnodes[k].n[d];
    }
  }

  const auto offset = nrs->fluid->fieldOffset;
  std::vector<dfloat> wrk(3 * offset, 0.0);
  double flux = 0, umax = 0, amax = 0;
  for (int k : actNodes) {
    double dn = 0, mag = 0;
    for (int d = 0; d < 3; d++) {
      wrk[bnodes[k].vol + d * offset] = u[3 * k + d];
      dn += u[3 * k + d] * bnodes[k].n[d];
      mag += u[3 * k + d] * u[3 * k + d];
    }
    flux += bnodes[k].w * dn;
    umax = std::max(umax, std::sqrt(mag));
  }
  for (auto v : din) amax = std::max(amax, std::abs(v));
  platform->app->bc->o_usrwrk.copyFrom(wrk.data(), 3 * offset);

  MPI_Allreduce(MPI_IN_PLACE, &flux, 1, MPI_DOUBLE, MPI_SUM, comm());
  MPI_Allreduce(MPI_IN_PLACE, &umax, 1, MPI_DOUBLE, MPI_MAX, comm());
  reply(0, {}, {flux, umax, shift, amax});
}

static void cmdMondf(const std::vector<int> &in)
{
  auto mesh = nrs->meshV;
  if (in.empty() || (int)in.size() != 1 + 2 * in[0]) return reply(1, {}, {}, "MONDF: expected [nm, (bid, kind) x nm]");
  std::vector<std::pair<int, int>> mon;
  for (int m = 0; m < in[0]; m++) {
    if (in[2 + 2 * m] < 0 || in[2 + 2 * m] > 1) return reply(1, {}, {}, "MONDF: kind is 0 (forces) or 1 (wall gradient)");
    mon.push_back({in[1 + 2 * m], in[2 + 2 * m]});
  }
  monitors = mon;
  o_monBid.clear();
  monArea.clear();
  auto o_one = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(mesh->Nlocal, 1.0, o_one);
  for (const auto &m : monitors) {
    std::vector<int> b = {m.first};
    o_monBid.push_back(platform->device.malloc<int>(1, b.data()));
    monArea.push_back(mesh->surfaceAreaMultiplyIntegrate(o_monBid.back(), o_one));
  }
  reply(0, {}, monArea);
}

// mean of grad(u_c) . n_in over the boundary, element-local gradient (Nek5000 gradm1 without averaging)
static std::array<double, 3> wallGradient(int bid, double area)
{
  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  std::vector<dfloat> lmm(mesh->Nlocal);
  mesh->o_LMM.copyTo(lmm.data(), mesh->Nlocal);
  std::array<double, 3> g = {0, 0, 0};
  for (int c = 0; c < 3; c++) {
    auto o_grad = opSEM::strongGrad(mesh, offset, nrs->fluid->o_U.slice(c * offset, mesh->Nlocal), false);
    std::vector<dfloat> gr(3 * offset);
    o_grad.copyTo(gr.data(), 3 * offset);
    o_grad.free();
    for (const auto &b : bnodes) {
      if (b.bid != bid) continue;
      double dn = 0;
      for (int d = 0; d < 3; d++) dn += gr[b.vol + d * offset] / lmm[b.vol] * b.n[d];
      g[c] += b.w * dn;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, g.data(), 3, MPI_DOUBLE, MPI_SUM, comm());
  for (auto &v : g) v /= area;
  return g;
}

static void recordStep(double time, int tstep)
{
  rows.insert(rows.end(), {static_cast<double>(tstep), time, static_cast<double>(nrs->dt[0]), computeCFL()});
  occa::memory o_Sij;
  for (size_t m = 0; m < monitors.size(); m++) {
    if (monitors[m].second == 0) {
      if (!o_Sij.isInitialized()) o_Sij = nrs->strainRate();
      auto f = nrs->aeroForces(o_monBid[m], o_Sij);
      const auto fv = f->viscous();
      const auto fp = f->pressure();
      rows.insert(rows.end(), {fv[0], fv[1], fv[2], fp[0], fp[1], fp[2]});
      delete f;
    } else {
      const auto g = wallGradient(monitors[m].first, monArea[m]);
      rows.insert(rows.end(), {g[0], g[1], g[2], 0.0, 0.0, 0.0});
    }
  }
  if (o_Sij.isInitialized()) o_Sij.free();
}

// ---------------------------------------------------------------- episode reset

// o_ADV (lagged advection terms) is private in fluidSolver_t; an explicit template instantiation may name
// private members, which gives access to it without modifying NekRS
template <typename Tag, typename Tag::type M> struct robber {
  friend typename Tag::type get(Tag) { return M; }
};
struct advTag {
  using type = occa::memory fluidSolver_t::*;
  friend type get(advTag);
};
template struct robber<advTag, &fluidSolver_t::o_ADV>;
static occa::memory &fluidADV()
{
  return (*nrs->fluid).*get(advTag());
}

// complete time-stepper state of the fluid: solution and pressure lags, explicit and advection term lags,
// convecting velocity (all levels, also with subcycling) and properties; together with dt[3] this reproduces
// the following steps (up to the initial guesses of the projection spaces, which are not reset)
static std::vector<occa::memory *> stateBuffers()
{
  auto f = &(*nrs->fluid);
  std::vector<occa::memory *> all = {&f->o_U, &f->o_P, &f->o_EXT, &fluidADV(), &f->o_relUrst, &f->o_prop};
  std::vector<occa::memory *> b;
  for (auto m : all) {
    if (m->isInitialized()) b.push_back(m);
  }
  return b;
}

static std::string resetUnsupported()
{
  if (!nrs->fluid) return "no fluid solver";
  if (nrs->scalar) return "cases with scalars are not supported yet";
  if (nrs->geom) return "moving meshes are not supported";
  if (nrs->neknek) return "neknek is not supported";
  return "";
}

static int historyLength()
{
  return std::max(nrs->fluid->o_coeffBDF.size(), nrs->fluid->o_coeffEXT.size());
}

struct snapshot {
  std::vector<occa::memory> buf;
  double dt[3];
  double time;
  int tstep;
};
static std::map<int, snapshot> snaps;

static std::string text(const std::vector<int> &in, size_t from)
{
  std::string s;
  for (size_t i = from; i < in.size(); i++) s.push_back(static_cast<char>(in[i]));
  return s;
}

static void cmdSnaps(const std::vector<int> &in)
{
  if (in.size() != 1 || in[0] < 0) return reply(1, {}, {}, "SNAPS expects [slot >= 0]");
  const auto err = resetUnsupported();
  if (!err.empty()) return reply(1, {}, {}, "SNAPS: " + err);
  if (curStep < historyLength()) {
    return reply(1, {}, {}, "SNAPS: the time stepper needs " + std::to_string(historyLength()) +
                            " steps of history, now at step " + std::to_string(curStep) + " (run STEPS first)");
  }
  auto &s = snaps[in[0]];
  const auto bufs = stateBuffers();
  s.buf.resize(bufs.size());
  double bytes = 0;
  for (size_t i = 0; i < bufs.size(); i++) {
    if (!s.buf[i].isInitialized() || s.buf[i].size() != bufs[i]->size()) {
      s.buf[i] = platform->device.malloc<dfloat>(bufs[i]->size());
    }
    s.buf[i].copyFrom(*bufs[i]);
    bytes += bufs[i]->size() * sizeof(dfloat);
  }
  for (int l = 0; l < 3; l++) s.dt[l] = nrs->dt[l];
  s.time = curTime;
  s.tstep = curStep;
  MPI_Allreduce(MPI_IN_PLACE, &bytes, 1, MPI_DOUBLE, MPI_SUM, comm());
  reply(0, {static_cast<int>(snaps.size())}, {curTime, bytes});
}

static void restoreSnapshot(const snapshot &s)
{
  const auto bufs = stateBuffers();
  for (size_t i = 0; i < bufs.size(); i++) bufs[i]->copyFrom(s.buf[i]);
  for (int l = 0; l < 3; l++) nrs->dt[l] = s.dt[l];
}

// field file (fld/restart) as initial condition: load u, p into level 0 and rebuild a frozen history, i.e.
// all lags equal the loaded state; the first step then acts like BDF1/EXT1 with dt' = dt b0^-1 (settle steps)
static std::string loadFieldFile(const std::string &spec)
{
  auto f = &(*nrs->fluid);
  if (nrs->advectionSubcycingSteps) return "field files with advection subcycling are not supported, use snapshots";
  const auto fileName = spec.substr(0, spec.find('+'));
  int ok = 1;
  if (rank() == 0) ok = std::ifstream(fileName).good();
  MPI_Bcast(&ok, 1, MPI_INT, 0, comm());
  if (!ok) return "cannot open " + fileName;

  nrs->restartFromFiles({spec + "+U+P"});

  const auto offU = f->fieldOffsetSum, offP = f->fieldOffset;
  for (dlong l = 1; l < static_cast<dlong>(f->o_U.size() / offU); l++) f->o_U.copyFrom(f->o_U, offU, l * offU, 0);
  for (dlong l = 1; l < static_cast<dlong>(f->o_P.size() / offP); l++) f->o_P.copyFrom(f->o_P, offP, l * offP, 0);

  // explicit and advection terms of the loaded state in all lag levels (as initInnerStep does for level 0)
  f->extrapolateSolution(); // equal lags: o_Ue = o_U
  nrs->computeUrst();
  platform->linAlg->fill(f->fieldOffsetSum, 0.0, f->o_EXT);
  if (nrs->userSource) nrs->userSource(curTime);
  if (platform->options.compareArgs("EQUATION TYPE", "NAVIERSTOKES")) f->makeAdvection(curTime, curStep);
  f->makeExplicit(curTime, curStep);
  auto &o_ADV = fluidADV();
  for (dlong l = 1; l < static_cast<dlong>(f->o_EXT.size() / offU); l++) {
    f->o_EXT.copyFrom(f->o_EXT, offU, l * offU, 0);
    if (o_ADV.isInitialized()) o_ADV.copyFrom(o_ADV, offU, l * offU, 0);
  }
  return "";
}

// RESET ints [0, slot]: restore a snapshot; [1, chars of "file[+int][+time=..]"]: load a field file.
// The solver clock and step counter keep running; the actuation is zeroed.
static void cmdReset(const std::vector<int> &in)
{
  if (in.size() < 2 || in[0] < 0 || in[0] > 1) return reply(1, {}, {}, "RESET expects [0, slot] or [1, file name]");
  const auto err = resetUnsupported();
  if (!err.empty()) return reply(1, {}, {}, "RESET: " + err);
  double t0 = curTime;
  if (in[0] == 0) {
    const auto it = snaps.find(in[1]);
    if (it == snaps.end()) return reply(1, {}, {}, "RESET: no snapshot in slot " + std::to_string(in[1]));
    restoreSnapshot(it->second);
    t0 = it->second.time;
  } else {
    const auto e = loadFieldFile(text(in, 1));
    if (!e.empty()) return reply(1, {}, {}, "RESET: " + e);
    platform->options.getArgs("START TIME", t0);
  }
  platform->linAlg->fill(3 * nrs->fluid->fieldOffset, 0.0, platform->app->bc->o_usrwrk);
  if (rank() == 0) {
    std::cout << "[HGNEK] RESET: " << (in[0] == 0 ? "snapshot " + std::to_string(in[1]) : text(in, 1))
              << " (state time " << t0 << ") at solver time " << curTime << std::endl;
  }
  reply(0, {}, {curTime, t0});
}

// SNAPF ints [op (0 write, 1 read), slot, chars of the path]: one binary file per rank, path_r<rank>
static void cmdSnapf(const std::vector<int> &in)
{
  if (in.size() < 3 || in[0] < 0 || in[0] > 1) return reply(1, {}, {}, "SNAPF expects [op, slot, path]");
  const auto err = resetUnsupported();
  if (!err.empty()) return reply(1, {}, {}, "SNAPF: " + err);
  auto mesh = nrs->meshV;
  const int op = in[0], slot = in[1];
  std::ostringstream fn;
  fn << text(in, 2) << "_r" << std::setw(5) << std::setfill('0') << rank();

  // geometry fingerprint of the local partition
  const auto [x, y, z] = mesh->xyzHost();
  double geo = 0;
  for (dlong n = 0; n < mesh->Nlocal; n++) geo += x[n] + 2 * y[n] + 3 * z[n];
  const auto bufs = stateBuffers();
  std::vector<int64_t> head = {0x48474e53, 1, platform->comm.mpiCommSize(), rank(), mesh->Nelements, mesh->Nlocal,
                               nrs->fluid->fieldOffset, static_cast<int64_t>(sizeof(dfloat)), static_cast<int64_t>(bufs.size())};
  for (auto b : bufs) head.push_back(b->size());

  std::string e;
  snapshot loaded;
  if (op == 0) {
    const auto it = snaps.find(slot);
    if (it == snaps.end()) {
      e = "no snapshot in slot " + std::to_string(slot);
    } else {
      std::ofstream out(fn.str(), std::ios::binary);
      const auto &s = it->second;
      const double meta[6] = {geo, s.dt[0], s.dt[1], s.dt[2], s.time, static_cast<double>(s.tstep)};
      out.write(reinterpret_cast<const char *>(head.data()), head.size() * sizeof(int64_t));
      out.write(reinterpret_cast<const char *>(meta), sizeof(meta));
      for (const auto &m : s.buf) {
        std::vector<dfloat> h(m.size());
        m.copyTo(h.data());
        out.write(reinterpret_cast<const char *>(h.data()), h.size() * sizeof(dfloat));
      }
      if (!out) e = "write failed: " + fn.str();
    }
  } else {
    std::ifstream inp(fn.str(), std::ios::binary);
    std::vector<int64_t> fh(head.size());
    double meta[6];
    inp.read(reinterpret_cast<char *>(fh.data()), fh.size() * sizeof(int64_t));
    inp.read(reinterpret_cast<char *>(meta), sizeof(meta));
    if (!inp) {
      e = "cannot read " + fn.str();
    } else if (fh != head) {
      e = fn.str() + " was written for a different partition, polynomial order or solver configuration";
    } else if (std::abs(meta[0] - geo) > 1e-10 * (std::abs(geo) + mesh->Nlocal)) {
      e = fn.str() + " was written for a different mesh";
    } else {
      auto &s = loaded;
      for (auto b : bufs) {
        std::vector<dfloat> h(b->size());
        inp.read(reinterpret_cast<char *>(h.data()), h.size() * sizeof(dfloat));
        s.buf.push_back(platform->device.malloc<dfloat>(h.size(), h.data()));
      }
      for (int l = 0; l < 3; l++) s.dt[l] = meta[1 + l];
      s.time = meta[4];
      s.tstep = static_cast<int>(meta[5]);
      if (!inp) e = "truncated file " + fn.str();
    }
  }

  // every rank must agree, the first error wins
  int bad = !e.empty(), any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, comm());
  if (any) {
    int len = e.size(), src = bad ? rank() : platform->comm.mpiCommSize();
    MPI_Allreduce(MPI_IN_PLACE, &src, 1, MPI_INT, MPI_MIN, comm());
    MPI_Bcast(&len, 1, MPI_INT, src, comm());
    e.resize(len);
    MPI_Bcast(e.data(), len, MPI_CHAR, src, comm());
    return reply(1, {}, {}, "SNAPF: " + e);
  }
  if (op == 1) snaps[slot] = std::move(loaded); // only once every rank has read its part
  const auto &s = snaps[slot];
  reply(0, {s.tstep}, {s.time});
}

static void terminate()
{
  if (rank() == 0) std::cout << "[HGNEK] TERMN" << std::endl;
  MPI_Barrier(comm());
  std::cout << std::flush;
  fflush(stdout);
  fflush(stderr);
  MPI_Finalize();
  // _Exit skips the NekRS static destructors, which would call MPI after MPI_Finalize
  std::_Exit(EXIT_SUCCESS);
}

// serve requests until the controller asks to advance the solver
static void serve()
{
  while (true) {
    char request[6] = {0};
    if (rank() == 0) MPI_Recv(request, 5, MPI_CHAR, 0, tagCmd, inter, MPI_STATUS_IGNORE);
    MPI_Bcast(request, 5, MPI_CHAR, 0, comm());
    const std::string req(request, 5);
    if (req == "TERMN") terminate();

    const auto in = recvPayload<int>(tagInt, MPI_INT);
    const auto din = recvPayload<double>(tagDbl, MPI_DOUBLE);

    if (req == "HELLO") {
      cmdHello();
    } else if (req == "BNDRY") {
      cmdBndry(in);
    } else if (req == "PRBDF") {
      cmdPrbdf(in, din);
    } else if (req == "PROBE") {
      cmdProbe();
    } else if (req == "ACTDF") {
      cmdActdf(in, din);
    } else if (req == "ACTUA") {
      cmdActua(din);
    } else if (req == "MONDF") {
      cmdMondf(in);
    } else if (req == "SNAPS") {
      cmdSnaps(in);
    } else if (req == "RESET") {
      cmdReset(in);
    } else if (req == "SNAPF") {
      cmdSnapf(in);
    } else if (req == "STEPS") {
      if (in.size() != 1 || in[0] < 1) {
        reply(1, {}, {}, "STEPS expects [n >= 1]");
        continue;
      }
      stepsLeft = in[0];
      rows.clear();
      return;
    } else {
      reply(1, {}, {}, "unknown request <" + req + ">");
    }
  }
}

static void setup0()
{
  std::string s;
  if (platform->par->extract("casedata", "hydrogymboundaries", s)) {
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream is(s);
    int b;
    while (is >> b) allowed.push_back(b);
  }
}

static void setup()
{
  // three velocity components per volume node, zero until the first ACTUA
  platform->app->bc->o_usrwrk.resize(3 * nrs->fluid->fieldOffset);
  platform->linAlg->fill(3 * nrs->fluid->fieldOffset, 0.0, platform->app->bc->o_usrwrk);
}

static void executeStep(double time, int tstep)
{
  curTime = time;
  curStep = tstep;
  if (tstep == 0) {
    nekrsCheck(allowed.empty(), comm(), EXIT_FAILURE, "%s\n", "set [CASEDATA] hydrogymBoundaries in the par file");
    setupGeometry();
    MPI_Intercomm_create(comm(), 0, MPI_COMM_WORLD, 0, 99, &inter);
    if (rank() == 0) std::cout << "[HGNEK] intercommunicator established, serving requests" << std::endl;
    serve();
    return;
  }
  if (stepsLeft == 0) return; // not reached: the solver only advances inside STEPS
  recordStep(time, tstep);
  if (--stepsLeft == 0) {
    const int rowLen = 4 + 6 * monitors.size();
    reply(0, {static_cast<int>(rows.size() / rowLen), rowLen}, rows);
    serve();
  }
}

} // namespace hgnek

#endif
