package ai.nadar;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
public class VectorMulOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder BO = ByteOrder.nativeOrder();
    private static final int TILE = 64;
    // 64³

    private VectorMulOps() {}

    /**
     * C = A @ B  — dispatches to Java tiled or BLIS native path.
     * A: (M, K), B: (K, N), C: (M, N) — all FLOAT32, contiguous.
     */
    public static void matmul(NDArray a, NDArray b, NDArray c) {
        checkMatmul(a, b, c);
        long M = a.shape[0], K = a.shape[1], N = b.shape[1];
        // TODO: BLIS native path — for now fall through to tiled
        // My design here is to check the dimensions of the tensor and if its over a certain threshold (BLIS_THRESHOLD)
        // then offload it to the GPU / Native BLIS binding else use the Vector API for cpu computation.
        tiledMatmul(a, b, c, (int) M, (int) K, (int) N);
    }

    private static void tiledMatmul(NDArray a, NDArray b, NDArray c,
                                    int M, int K, int N) {
        int step = SPECIES.length();

        for (int i0 = 0; i0 < M; i0 += TILE) {
            int iEnd = Math.min(i0 + TILE, M);
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int kEnd = Math.min(k0 + TILE, K);
                for (int j0 = 0; j0 < N; j0 += TILE) {
                    int jEnd = Math.min(j0 + TILE, N);

                    // Inner kernel — process one tile
                    for (int i = i0; i < iEnd; i++) {
                        for (int k = k0; k < kEnd; k++) {
                            float aVal = a.segment.get(ValueLayout.JAVA_FLOAT,
                                    a.offset + ((long) i * K + k) * Float.BYTES);
                            FloatVector va = FloatVector.broadcast(SPECIES, aVal);

                            int j = j0;
                            int jBound = j0 + SPECIES.loopBound(jEnd - j0);
                            for (; j < jBound; j += step) {
                                long bOff = b.offset + ((long) k * N + j) * Float.BYTES;
                                long cOff = c.offset + ((long) i * N + j) * Float.BYTES;
                                FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b.segment, bOff, BO);
                                FloatVector vc = FloatVector.fromMemorySegment(SPECIES, c.segment, cOff, BO);
                                va.fma(vb, vc).intoMemorySegment(c.segment, cOff, BO);
                            }
                            // Scalar tail
                            for (; j < jEnd; j++) {
                                long bOff = b.offset + ((long) k * N + j) * Float.BYTES;
                                long cOff = c.offset + ((long) i * N + j) * Float.BYTES;
                                float prev = c.segment.get(ValueLayout.JAVA_FLOAT, cOff);
                                c.segment.set(ValueLayout.JAVA_FLOAT, cOff,
                                        prev + aVal * b.segment.get(ValueLayout.JAVA_FLOAT, bOff));
                            }
                        }
                    }
                }
            }
        }
    }

    private static void checkMatmul(NDArray a, NDArray b, NDArray c) {
        if (a.dtype != DType.FLOAT32 || b.dtype != DType.FLOAT32 || c.dtype != DType.FLOAT32)
            throw new IllegalArgumentException("BlisEngine requires 2dArrays for now...");
        if (a.shape.length != 2 || b.shape.length != 2 || c.shape.length != 2)
            throw new IllegalArgumentException("BlisEngine requires 2D arrays :(");
        if (a.shape[1] != b.shape[0])
            throw new IllegalArgumentException("Shape mismatch: A cols " + a.shape[1] + " != B rows " + b.shape[0]);
        if (c.shape[0] != a.shape[0] || c.shape[1] != b.shape[1])
            throw new IllegalArgumentException("Output shape mismatch");
        if (!a.isContiguous() || !b.isContiguous() || !c.isContiguous())
            throw new UnsupportedOperationException("Non-contiguous arrays not supported");
    }
}