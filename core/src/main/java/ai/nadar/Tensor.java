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
    NDArray ndarray() {
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
    public void scale_(float v) {
        VectorOps.scaleInPlace(this.array, v);
    }
}
