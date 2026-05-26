package ai.nadar;

import java.lang.foreign.Arena;
import java.util.Arrays;

// TODO: Improve the docstring to mention all the methods provided by the Tensor class. (also the other docstrings)

/**
 * Public API facade for tensors.
 * This API is the wrapper for all the internal mechanisms.
 * The design philosophy of typical idiomatic ML/Numeric computation APIs should be followed while developing this.
 */
public class Tensor implements AutoCloseable {

    private final NDArray array;
    private final Arena arena;

    private Tensor(NDArray array, Arena arena) {
        this.array = array;
        this.arena = arena;
    }

    /**
     * Creates a tensor of zeros with the given dtype and shape.
     */
    public static Tensor zeros(DType dtype, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.zeros(dtype, arena, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Creates a tensor of ones with the given dtype and shape.
     */
    public static Tensor ones(DType dtype, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.ones(dtype, arena, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Creates a tensor filled with {@code value} with the given dtype and shape.
     */
    public static Tensor fill(DType dtype, float value, long... shape) {
        Arena arena = Arena.ofConfined();
        NDArray arr = NDArray.fill(dtype, arena, value, shape);
        return new Tensor(arr, arena);
    }

    /**
     * Addition operation
     * return
     */
    public Tensor add(Tensor other) {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.add(this.array, other.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Returns a defensive copy of the shape array.
     *
     * @return Tensor (consisting of the sum of the values).
     */
    public long[] shape() {
        return Arrays.copyOf(array.shape, array.shape.length);
    }

    public DType dtype() {
        return array.dtype;
    }

    /**
     * Number of dimensions.
     */
    public int ndims() {
        return array.shape.length;
    }

    /**
     * Total number of elements.
     */
    public long size() {
        return NDArray.elementCount(array.shape);
    }

    public float getFloat(long... indices) {
        return array.getFloat(indices);
    }

    public void setFloat(float v, long... indices) {
        array.setFloat(v, indices);
    }

    // Package-private NDArray access (for ops layer)
    public NDArray ndarray() {
        return array;
    }

    @Override
    public String toString() {
        return array.toString();
    }

    /**
     * Closes the underlying arena, freeing all off-heap memory for this tensor.
     */
    @Override
    public void close() {
        arena.close();
    }

    // TODO: put them in a better spot (lazy so just adding them in the end - low priority)
    // This arena is never closed (it should be this needs fixing)
    public void relu_(Tensor other) {
        Arena outArena = Arena.ofConfined();
        VectorOps.reluInPlace(other.ndarray());
    }

    /**
     * Allocating scale operation. Returns a new Tensor.
     */
    public void scale(float v) {
        VectorOps.scaleInPlace(this.array, v);
//        Arena outArena = Arena.ofConfined();
//        NDArray outArr = VectorOps.scale(this.array, v, outArena);
    }

    /**
     * Allocating ReLU operation. Returns a new Tensor.
     */
    public Tensor relu() {
        Arena outArena = Arena.ofConfined();
        NDArray outArr = VectorOps.relu(this.array, outArena);
        return new Tensor(outArr, outArena);
    }

    /**
     * Reduction sum operation. Returns a primitive float.
     */
    public float sum() {
        return VectorOps.sum(this.array);
    }

    /**
     * In-Place ReLU operation. Mutates this tensor directly.
     * Notice the method name ends with an underscore (idiomatic for in-place).
     */
    public void relu_() {
        // No new Arena! We modify our own memory directly.
        VectorOps.reluInPlace(this.array);
    }

    /**
     * In-Place Scale operation. Mutates this tensor directly.
     */
    // TODO: Decide whether to keep this or stay() - (without the underscore)
    public void scale_(float v) {
        VectorOps.scaleInPlace(this.array, v);
    }

/**
     * Matrix multiplication. Returns a new Tensor.
     * <p>
     * Supports batched matmul with broadcasting for any dimensionality:
     * <ul>
     *   <li>2D:       (M,K) @ (K,N) → (M,N)
     *   <li>Batched:  (...,M,K) @ (...,K,N) → (...,M,N)
     *   <li>1D @ 1D:  (K,)   @ (K,)   → (1,)       — dot product
     *   <li>1D @ ND:  (K,)   @ (...,K,N) → (...,N)
     *   <li>ND @ 1D:  (...,M,K) @ (K,)   → (...,M)
     * </ul>
     */
    public Tensor matmul(Tensor other) {
        int aRank = this.ndims();
        int bRank = other.ndims();

        // Compute output shape
        long[] outShape;
        if (aRank == 1 && bRank == 1) {
            outShape = new long[]{1};
        } else if (aRank == 1) {
            // (K,) @ (...,K,N) → (...,N)
            long K = this.shape()[0];
            long[] bShape = other.shape();
            if (K != bShape[bRank - 2])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + K + " vs " + bShape[bRank - 2]);
            outShape = new long[bRank - 1];
            System.arraycopy(bShape, 0, outShape, 0, bRank - 2);
            outShape[bRank - 2] = bShape[bRank - 1];
        } else if (bRank == 1) {
            // (...,M,K) @ (K,) → (...,M)
            long K = other.shape()[0];
            long[] aShape = this.shape();
            if (K != aShape[aRank - 1])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + aShape[aRank - 1] + " vs " + K);
            outShape = new long[aRank - 1];
            System.arraycopy(aShape, 0, outShape, 0, aRank - 2);
            outShape[aRank - 2] = aShape[aRank - 2];
        } else {
            // Both ≥ 2D
            long K = this.shape()[aRank - 1];
            long M = this.shape()[aRank - 2];
            long N = other.shape()[bRank - 1];
            if (K != other.shape()[bRank - 2])
                throw new IllegalArgumentException("Inner dimensions mismatch: "
                        + K + " != " + other.shape()[bRank - 2]);

            int aBatchRank = aRank - 2;
            int bBatchRank = bRank - 2;
            int batchRank = Math.max(aBatchRank, bBatchRank);

            outShape = new long[batchRank + 2];
            for (int i = 0; i < batchRank; i++) {
                long da = i < aBatchRank
                    ? this.shape()[aBatchRank - 1 - i] : 1;
                long db = i < bBatchRank
                    ? other.shape()[bBatchRank - 1 - i] : 1;
                if (da != 1 && db != 1 && da != db)
                    throw new IllegalArgumentException(
                        "Incompatible batch dimensions at axis "
                        + (batchRank - 1 - i) + ": " + da + " vs " + db);
                outShape[batchRank - 1 - i] = Math.max(da, db);
            }
            outShape[batchRank]     = M;
            outShape[batchRank + 1] = N;
        }

        Arena outArena = Arena.ofConfined();
        NDArray out = NDArray.zeros(DType.FLOAT32, outArena, outShape);
        VectorMulOps.matmul(this.array, other.ndarray(), out);
        return new Tensor(out, outArena);
    }

    public String mean() {
        return String.valueOf(VectorOps.mean(this.array));
    }
}
