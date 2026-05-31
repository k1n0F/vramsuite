from __future__ import annotations

from dataclasses import dataclass

from vramsuite.backends.cublaslt.descriptors import MatrixLayout
from vramsuite.backends.cublaslt.enums import CUDA_R_32F
from vramsuite.backends.cublaslt.runtime import CublasLtRuntime



@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        



@dataclass(frozen=True)
class MatrixLayoutSpec:
    name: str
    rows: int
    cols: int
    ld: int
    data_type: int = CUDA_R_32F

    def validate(self) -> None:
        if self.rows <= 0:
            raise ValueError(f"{self.name}.rows must be positive")
        if self.cols <= 0:
            raise ValueError(f"{self.name}.cols must be positive")
        if self.ld <= 0:
            raise ValueError(f"{self.name}.ld must be positive")
        if self.ld < self.rows:
            raise ValueError(
                f"{self.name}.ld must be >= rows for column-major layout"
            )
        
    def create(self, runtime: CublasLtRuntime) -> MatrixLayout:
        self.validate()

        return MatrixLayout(
            runtime,
            data_type=self.data_type,
            rows=self.rows,
            cols=self.cols,
            ld=self.ld,
        )
    

@dataclass(frozen=True)
class MatmulLayoutSpec:
    shape: GemmShape
    a: MatrixLayoutSpec
    b: MatrixLayoutSpec
    c: MatrixLayoutSpec
    d: MatrixLayoutSpec

    @classmethod
    def fp32_column_major_nn(cls, *, m: int, n: int, k: int) -> "MatmulLayoutSpec":
        """
        Column-major NN GEMM:

            D = A @ B

        A: m x k
        B: k x n
        C: m x n
        D: m x n

        ld for column-major is at least number of rows.
        """

        shape = GemmShape(m=m, n=n, k=k)
        shape.validate()

        return cls(
            shape=shape,
            a=MatrixLayoutSpec(
                name="A",
                rows=m,
                cols=k,
                ld=m,
                data_type=CUDA_R_32F,
            ),
            b=MatrixLayoutSpec(
                name="B",
                rows=k,
                cols=n,
                ld=k,
                data_type=CUDA_R_32F,
            ),
            c=MatrixLayoutSpec(
                name="C",
                rows=m,
                cols=n,
                ld=m,
                data_type=CUDA_R_32F,
            ),
            d=MatrixLayoutSpec(
                name="D",
                rows=m,
                cols=n,
                ld=m,
                data_type=CUDA_R_32F,
            ),
        )
    
    def validate(self) -> None:
        self.shape.validate()
        self.a.validate()
        self.b.validate()
        self.c.validate()
        self.d.validate()