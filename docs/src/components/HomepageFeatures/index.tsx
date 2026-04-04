import type {ReactNode} from 'react';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Diverse Environments',
    description: (
      <>
        88 pre-configured environments across 6 CFD solvers. Flow environments
        are supported by multiple backends: Finite Element (Firedrake),
        Lattice Boltzmann (MAIA LBM), Finite Volume (MAIA FV), Spectral
        Element (NEK5000), and fully Differentiable solvers (JAX-Fluids)
      </>
    ),
  },
  {
    title: 'Scalable Implementation',
    description: (
      <>
        Highly optimized GPU & CPU backends for efficient RL deployment
        ranging from local workstations to exascale HPC systems
      </>
    ),
  },
  {
    title: 'Research Ready',
    description: (
      <>
        Includes checkpoints, observation strategies, and reward formulations
        managed by a complementary HuggingFace repository. Gymnasium-compatible
        API works with Stable-Baselines3, RLlib, and other RL libraries
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

type ExamplePanel = {
  title: string;
  description: ReactNode;
  img: string;
};

const examplePanels: ExamplePanel[] = [
  {
    title: 'NACA 0012 Airfoil',
    description: (
      <>
        Zero shot evaluation at Re<sub>c</sub> = 200,000. Drag reduction trained
        on channel flow of Re<sub>τ</sub> = 206, and evaluated on a large scale
        airfoil.
      </>
    ),
    img: '@site/static/img/La2_200k_animation-2.gif',
  },
  {
    title: 'Fluidic Pinball',
    description: (
      <>
        Multi-body wake interactions at Re = 30 - 150. Coordinated control of
        three cylinders demonstrates chaos suppression.
      </>
    ),
    img: '@site/static/img/pinball.gif',
  },
  {
    title: 'Turbulent Boundary Layer',
    description: (
      <>
        Reinforcement learning agent performs net power savings with travelling
        wave control for Re<sub>τ</sub> = 200, 1550, and 2200
      </>
    ),
    img: '@site/static/img/tbl_lambda2_sliced-ezgif.com-video-to-gif-converter.gif',
  },
  {
    title: 'Cylinder',
    description: (
      <>
        2D & 3D flows at Re = 100 - 3,900. Drag reduction via rotation and
        jet actuation. Achieves more than 20% drag reduction.
      </>
    ),
    img: '@site/static/img/cylinder.gif',
  },
  {
    title: 'Turbulent Channel Flow (Differentiable)',
    description: (
      <>
        Wall-shear stress reduction at Re<sub>τ</sub> = 180. Gradient-enhanced
        training with JAX for efficient optimization.
      </>
    ),
    img: '@site/static/img/controlled_animation.gif',
  },
  {
    title: 'NACA 0012 Airfoil',
    description: (
      <>
        Gust mitigation at Re = 100 - 50,000. Transverse gust encounters with
        load alleviation strategies.
      </>
    ),
    img: '@site/static/img/naca.gif',
  },
  {
    title: 'Kolmogorov Flow (Differentiable)',
    description: (
      <>
        Extreme event mitigation in 2D turbulence. Control energy bursts
        and enhance mixing efficiency.
      </>
    ),
    img: '@site/static/img/kolmogorov.gif',
  },
  {
    title: 'Open Cavity Flow',
    description: (
      <>
        Shear layer stabilization at Re = 4,140 - 7,500.
        Acoustic feedback disruption through leading-edge control.
      </>
    ),
    img: '@site/static/img/cavity.gif',
  },
];

function ExamplePanelCard({item}: {item: ExamplePanel}) {
  const src = useBaseUrl(item.img);
  return (
    <li className={styles.examplePanel}>
      <div className={styles.examplePanelInner}>
        <div className={styles.exampleThumbWrap}>
          <img
            className={styles.exampleThumb}
            src={src}
            alt={item.title}
            loading="lazy"
            decoding="async"
            draggable={false}
          />
        </div>
        <div className={styles.exampleText}>
          <span className={styles.exampleTitle}>{item.title}</span>
          <p className={styles.exampleDescription}>{item.description}</p>
        </div>
      </div>
    </li>
  );
}

function ExampleShowcase(): ReactNode {
  return (
    <div className={styles.examplesShowcase}>
      <p className={styles.examplesLead}>
        Join researchers training flow-control agents on canonical benchmarks in
        minutes.
      </p>
      <ul className={styles.examplesGrid} role="list">
        {examplePanels.map((item) => (
          <ExamplePanelCard key={item.img} item={item} />
        ))}
      </ul>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.aboutIntro}>
          <Heading as="h3">About the Project</Heading>
          <p className={styles.aboutText}>
            HydroGym is a transparent, extensible & comprehensive platform
            for applying reinforcement learning to fluid dynamics flow control.
            With environments ranging from canonical benchmarks to turbulent
            flows, HydroGym provides a standardized Gymnasium-compatible
            interface for training RL agents on challenging CFD problems.
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
        <ExampleShowcase />
      </div>
    </section>
  );
}
