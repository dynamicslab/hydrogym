# Ray Cluster Setup

## Manual Ray Cluster

Especially for a manual cluster we need to consider a few basic principles:

- All machines need to have **the same** basic setup with the package pre-installed, and all dependencies present

We can then initialize the headnode

```bash
ray start --head
```

and subsequently attach all the worker nodes

```bash
ray start --address=IP_OF_THE_HEAD_NODE
```

> If a fast interconnect is available e.g. higher-bandwidth ethernet, Omnipath, Infiniband, etc. consider using the custom IPs of that interconnect to force the usage of the faster connections.

With the cluster now initialized we can use it in any of our scripts With

```python
import ray
ray.init(address=IP_OF_THE_HEAD_NODE)
```


## Ray Cluster on Kubernetes

Ray can easily be installed on Kubernetes with [Helm](https://helm.sh) using the Ray Kubernetes Operator, the detailed documentation for which can be found in [Ray's documentation](https://docs.ray.io/en/latest/cluster/kubernetes.html).

> On clouds like AWS, Azure, and GCP this is entirely straightforward and automated away.

## Ray Cluster on Slurm

Ray on Slurm is currently still experimental in its support, but the progress and current approach can be viewed in [Ray's documentation](https://docs.ray.io/en/latest/cluster/slurm.html). The biggest difference here coming from the different way Ray and Slurm treat 

1. Ports binding
2. IP binding

But there exist example [launch script](https://docs.ray.io/en/latest/cluster/examples/slurm-launch.html#slurm-launch), and [templated launch scripts](https://docs.ray.io/en/latest/cluster/examples/slurm-template.html#slurm-template).

