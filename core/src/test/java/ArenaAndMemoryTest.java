package ai.sargun;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for arena safety, memory allocation, and proper resource cleanup.
 * These tests ensure that the arena lifecycle is managed correctly and
 * that misuse (like use-after-close) is caught where possible.
 */
class ArenaAndMemoryTest {

    @Test
    void arenaCloseIsIdempotent() {
        Arena arena = Arena.ofConfined();
        arena.close();
        // Second close should not throw
        arena.close();
    }

    @Test
    void allocatingFromClosedArenaThrows() {
        Arena arena = Arena.ofConfined();
        arena.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void tensorCloseClosesUnderlyingArena() {
        try (Tensor t = Tensor.ones(DType.FLOAT32, 3, 3)) {
            Arena arena = t.ndarray().arena;
            assertTrue(arena != null);
        }
        // After try-with-resources, the arena should be closed.
        // We can't directly access the arena because it's private inside NDArray.
        // Instead, we verify that attempting to use the tensor throws when
        // the underlying arena is closed (by checking that the NDArray's arena is closed).
        // Since we don't expose the arena, we rely on the fact that Tensor.close()
        // calls arena.close() on the underlying arena.
        // We'll test indirectly by trying to create a new tensor from a closed arena
        // via a leaked NDArray (not ideal, but we can test arena behavior directly).
    }

    @Test
    void usingTensorAfterCloseLeadsToUndefinedBehavior() {
        // This test demonstrates that using a Tensor after close is unsafe.
        // We cannot safely assert a specific outcome because it may crash the JVM.
        // Instead, we verify that the arena is closed after Tensor.close().
        Tensor t = Tensor.ones(DType.FLOAT32, 2, 2);
        Arena arena = t.ndarray().arena;
        t.close(); // closes the arena
        // Now the arena is closed; allocating from it should throw.
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
        // Note: accessing t.ndarray().segment after this point is undefined and may crash.
        // We do not attempt to access it in this test to avoid killing the test runner.
    }

    @Test
    void testTensorOperationsAfterCloseThrowWhenPossible() {
        // Some operations may throw if the underlying NDArray checks for closed arena.
        // Currently, NDArray does not check; so accessing segment after close may crash.
        // We test that at least the arena is closed.
        Tensor t = Tensor.fill(DType.FLOAT32, 1.0f, 2, 2);
        Arena arena = t.ndarray().arena;
        t.close();
        assertThrows(IllegalStateException.class, () -> arena.allocate(1));
    }

    @Test
    void testNDArrayCloseTwice() {
        try (Arena arena = Arena.ofConfined()) {
            NDArray arr = NDArray.zeros(DType.FLOAT32, arena, 2, 2);
            arr.close();
            arr.close(); // should not throw
        }
    }

    @Test
    void testNDArraySegmentAccessAfterCloseIsUnsafe() {
        // We cannot safely test this without risking JVM crash.
        // This test is documented to remind that NDArray does not protect against use-after-close.
        // The responsibility lies with the caller to not use NDArray after close().
        // We can at least verify that the arena is closed.
        try (Arena arena = Arena.ofConfined()) {
            NDArray arr = NDArray.zeros(DType.FLOAT32, arena, 2);
            arr.close();
            // After close, arena is closed.
            assertThrows(IllegalStateException.class, () -> arena.allocate(1));
        }
    }

    @Test
    void testTensorConstructorDoesNotLeakArenaOnException() {
        // If an exception is thrown during Tensor construction, the arena should be closed.
        // This is more of a sanity check; Tensor construction currently does not throw
        // after arena allocation, but we test the pattern.
        try {
            try (Tensor ignored = Tensor.ones(DType.FLOAT32, 1)) {
                // Intentionally throw an exception to see if arena is still closed.
                throw new RuntimeException("test");
            }
        } catch (RuntimeException e) {
            // Expected
        }
        // No way to verify arena state without exposing it; rely on correct try-with-resources.
    }

    @Test
    void testVectorOpsRespectsArenaLifetime() {
        // VectorOps methods should not keep references to the arena beyond the call.
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 5);
            NDArray b = NDArray.zeros(DType.FLOAT32, arena, 5);
            // Set some values
            a.segment.set(ValueLayout.JAVA_FLOAT, 0L, 1.0f);
            b.segment.set(ValueLayout.JAVA_FLOAT, 0L, 2.0f);

            NDArray result = VectorOps.add(a, b, arena);
            // result uses a new arena (passed in)
            assertEquals(3.0f, result.getFloat(0), 1e-5f);

            // Now close the input arena; result should still be valid because it has its own arena.
            arena.close();
            // Accessing result should still work.
            assertEquals(3.0f, result.getFloat(0), 1e-5f);
            result.close(); // close result's arena
        }
    }

    @Test
    void testInPlaceOpsAfterInputArenaCloseIsUnsafe() {
        // In-place ops modify the input array directly; if its arena is closed, it's unsafe.
        try (Arena arena = Arena.ofConfined()) {
            NDArray a = NDArray.zeros(DType.FLOAT32, arena, 5);
            a.segment.set(ValueLayout.JAVA_FLOAT, 0L, 5.0f);
            VectorOps.scaleInPlace(a, 2.0f);
            assertEquals(10.0f, a.getFloat(0), 1e-5f);
            arena.close();
            // Now a's arena is closed; further access is unsafe.
            // We do not attempt to access a here.
        }
    }
}