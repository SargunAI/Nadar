package ai.nadar;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import static org.junit.jupiter.api.Assertions.*;

class VectorMulOpsTest {

    // Helper — fills NDArray from flat float array
    private void fill(NDArray arr, float... vals) {
        for (int i = 0; i < vals.length; i++)
            arr.setFloatAtIndex(vals[i], i);
    }

    @Test
    void testSmallMatmulCorrectness() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 2, 3);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 3, 2);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 2, 2);

            // A = [[1,2,3],[4,5,6]]
            fill(a, 1f, 2f, 3f, 4f, 5f, 6f);
            // B = [[7,8],[9,10],[11,12]]
            fill(b, 7f, 8f, 9f, 10f, 11f, 12f);

            VectorMulOps.matmul(a, b, c);

            // C[0,0] = 1*7 + 2*9 + 3*11 = 58
            assertEquals(58f,  c.getFloat(0, 0), 1e-4f);
            // C[0,1] = 1*8 + 2*10 + 3*12 = 64
            assertEquals(64f,  c.getFloat(0, 1), 1e-4f);
            // C[1,0] = 4*7 + 5*9 + 6*11 = 139
            assertEquals(139f, c.getFloat(1, 0), 1e-4f);
            // C[1,1] = 4*8 + 5*10 + 6*12 = 154
            assertEquals(154f, c.getFloat(1, 1), 1e-4f);
        }
    }

    @Test
    void testIdentityMatmul() {
        try (Arena arena = Arena.ofConfined()) {
            int N = 4;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray identity = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, N, N);

            // Fill A with 1..16
            for (int i = 0; i < N * N; i++) a.setFloatAtIndex(i + 1f, i);
            // Fill identity
            for (int i = 0; i < N; i++) identity.setFloat(1f, i, i);

            VectorMulOps.matmul(a, identity, c);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    assertEquals(a.getFloat(i, j), c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testNonMultipleOfSIMDWidth() {
        try (Arena arena = Arena.ofConfined()) {
            // 3x5 @ 5x3 = 3x3 — 5 and 3 are not multiples of 8 or 16
            int M = 3, K = 5, N = 3;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, M, K);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, K, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, M, N);

            // All ones — result should be K everywhere
            for (int i = 0; i < M * K; i++) a.setFloatAtIndex(1f, i);
            for (int i = 0; i < K * N; i++) b.setFloatAtIndex(1f, i);

            VectorMulOps.matmul(a, b, c);

            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    assertEquals((float) K, c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testLargerThanTileSize() {
        try (Arena arena = Arena.ofConfined()) {
            // 100x100 — crosses the 64-tile boundary
            int N = 100;
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, N, N);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, N, N);

            // A = all 1s, B = identity → C = A
            for (int i = 0; i < N * N; i++) a.setFloatAtIndex(1f, i);
            for (int i = 0; i < N; i++) b.setFloat(1f, i, i);

            VectorMulOps.matmul(a, b, c);

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    assertEquals(1f, c.getFloat(i, j), 1e-4f,
                            "Mismatch at [" + i + "," + j + "]");
        }
    }

    @Test
    void testShapeMismatchThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 5, 3); // K mismatch
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3, 3);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }

    @Test
    void testNon2DThrows() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 3, 4, 5); // 3D
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 5, 3);
            NDArray c = NDArray.zeros(DType.FLOAT32, arena, 3, 3);

            assertThrows(IllegalArgumentException.class,
                    () -> VectorMulOps.matmul(a, b, c));
        }
    }
}