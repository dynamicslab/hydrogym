---
layout: default
title: HydroGym
---


## Abstract

The modeling and control of fluid flows remains a grand challenge problem of the modern era, with potentially
transformative scientific, technological, and industrial impact. Better fluid flow control may enable drag reduction,
mixing enhancement, and noise reduction in domains as diverse as transportation, energy, and medicine.
Reinforcement learning has proven a highly effective control paradigm in complex environments such as robotics, games,
and protein folding, built on scalable platforms focused on standardized benchmark problems.
In contrast, progress in flow control is challenging because of the relative scarcity of such platforms and benchmarks in
conjunction with the computational complexity of fluid simulations, which require heavily parallelized, specialized solvers. 
As a result, the engineering burden of encapsulating these simulations in benchmark environments has proven to be a significant
barrier. In this paper we present a new solver-independent reinforcement learning platform for flow control called HydroGym,
connecting sophisticated flow control benchmark problems, a scalable runtime, and state-of-the-art reinforcement learning as well as
differentiable reinforcement learning algorithms. We have implemented an initial set of four fully validated non-differentiable
fluid flow environments, and one differentiable fluid flow environment that typify various flow control challenges, and which
are all evaluated with a broad set of reinforcement learning algorithms. This platform is designed to be scalable and
extensible, with computations scaling seamlessly from laptops to high-performance computing resources with a standardardized
reinforcement learning environment interface for ease of implementing new flow environments. Thus, HydroGym provides a
platform to simultaneously advance flow control research with state-of-the-art techniques in non-differentiable, as well as
differentiable reinforcement learning, while also providing sophisticated challenge environments to advance scientific
machine learning research.


## Construction of HydroGym


## Benchmarking


## Authors

<div style="display:table;margin: 0 auto;">
<div class="cards">
    <div class="card">
        <img src="paehler.jpg" height="300" width="300" alt="ludger" />
        <p style="text-align:center;"><a href="https://ludger.fyi">Ludger Paehler</a></p>
    </div>
    <div class="card">
        <img class="middle-img" src="callaham.jpeg" height="300" width="300" alt="jan" />
        <p style="text-align:center;"><a href="https://amath.washington.edu/people/jared-callaham">Jared Callaham</a></p>
    </div>
    <div class="card">
        <img src="ahnert.jpeg" height="300" width="300" alt="steffen" />
        <p style="text-align:center;"><a href="https://www.linkedin.com/in/samuel-ahnert-870287146">Samuel Ahnert</a></p>
    </div>
    <div class="card">
        <img src="adams.jpg" height="300" width="300" alt="klaus" />
        <p style="text-align:center;"><a href="https://www.epc.ed.tum.de/en/aer/members/cv/prof-adams/">Nikolaus A. Adams</a></p>
    </div>
    <div class="card">
        <img src="brunton.png" height="300" width="300" alt="steven" />
        <p style="text-align:center;"><a href="https://eigensteve.com">Steven L. Brunton</a></p>
    </div>
</div>
</div>

## Corresponding Authors

* Ludger Paehler ([ludger.paehler@tum.de](mailto:ludger.paehler@tum.de?subject=HydroGym))
* Steven L. Brunton ([sbrunton@uw.edu](mailto:sbrunton@uw.edu?subject=HydroGym))

## Citation

```bibtex
@inproceedings{paehler2024HydroGym,
}
```
