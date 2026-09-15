# Guess Right

PyTorch clone in C++ that implements automatic differentiation, optimizers and
some machine learning models that (hopefully) _guess right_.

## Linear Algebra

The core linear algebra module relies on the `NDArray` that only supports
scalar, vectors and matrices. Internally everything is a matrix but the API lets
the user do something like

```cpp
NDArray a = 2.0f;
```

The `NDArray` must be considered as a mathematical object only, not a data
structure like `numpy` arrays that allow for sorting, complex indexing,
broadcasting and so on.

The only way to use it is through valid linear algebra arithmetic operations.

## Automatic Differentiation

The main part of the library is the automatic differentiation tool that builds a
computational graph composed by _primitive_ operations.

For now the implementation of the graph relies on `std::shared_ptr` but this
causes a lot of overhead due to continue allocations and deallocations of nodes
and big matrices.

Future implementations should rely on some preallocated memory chunk.

## Usage
