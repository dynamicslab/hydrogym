---
sidebar_label: linalg
title: hydrogym.firedrake.utils.linalg
---

## LinearOperator Objects

```python
class LinearOperator(metaclass=abc.ABCMeta)
```

#### \_\_matmul\_\_

```python
@abc.abstractmethod
def __matmul__(v: fd.Function)
```

Return the matrix-vector product A @ v.

## DirectOperator Objects

```python
class DirectOperator(LinearOperator)
```

#### T

```python
@property
def T() -> InverseOperator
```

Return the adjoint operator.

This will solve the matrix pencil A^T @ f = M^T @ v0 for f.

## InverseOperator Objects

```python
@dataclasses.dataclass
class InverseOperator(LinearOperator)
```

A simple wrapper for the inverse of a matrix pencil.

Note that this object will own the output Function unless
`copy_output=True` is set.  This is memory-efficient, but could
lead to confusion if the output reference is modified. The Arnoldi
iteration algorithm is written so this isn&#x27;t a problem.

#### T

```python
@property
def T() -> InverseOperator
```

Return the adjoint operator.

This will solve the matrix pencil A^T @ f = M^T @ v0 for f.

#### \_\_matmul\_\_

```python
def __matmul__(v0)
```

Solve the matrix pencil A @ f = M @ v0 for f.

This is equivalent to the &quot;inverse iteration&quot; f = (A^{-1} @ M) @ v0

