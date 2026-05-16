package ai.nadar;

public class Demo {
    static void main() {
        System.out.println("--- Hardik Tensor Engine ---\n");

        // 1. ALLOCATION — three factory methods
        try (
                Tensor weights    = Tensor.fill(DType.FLOAT32, 0.5f, 1000, 1000);
                Tensor biases     = Tensor.ones(DType.FLOAT32, 1000, 1000);
                Tensor zeros      = Tensor.zeros(DType.FLOAT32, 1000, 1000)
        ) {
            System.out.println("Allocation");
            System.out.println("  weights : " + weights.shape()[0] + "x" + weights.shape()[1] + " filled with 0.5");
            System.out.println("  biases  : " + biases.shape()[0]  + "x" + biases.shape()[1]  + " filled with 1.0");
            System.out.println("  zeros   : " + zeros.shape()[0]   + "x" + zeros.shape()[1]   + " filled with 0.0");

            // 2. ALLOCATING OPS — add returns a new Tensor the caller owns
            try (Tensor preActivation = weights.add(biases)) {
                System.out.println("\nAllocating ops");
                System.out.println("  weights + biases[0,0]     = " + preActivation.getFloat(0, 0)); // 1.5

                // 3. IN-PLACE OPS — mutate existing memory, no allocation
                System.out.println("\nIn-place ops");
                preActivation.scale_(-1.0f);
                System.out.println("  after scale_(-1)[0,0]     = " + preActivation.getFloat(0, 0)); // -1.5
                preActivation.relu_();
                System.out.println("  after relu_()[0,0]        = " + preActivation.getFloat(0, 0)); // 0.0

                // 4. REDUCTIONS — return primitives, no allocation
                System.out.println("\nReductions");
                Tensor sampleData = Tensor.fill(DType.FLOAT32, 2.0f, 1000, 1000);
                System.out.println("  sum of 1M x 2.0           = " + sampleData.sum());   // 2_000_000.0
                sampleData.close();
            }

            // 5. ELEMENT ACCESS — get and set
            System.out.println("\nElement access");
            try (Tensor t = Tensor.zeros(DType.FLOAT32, 4, 4)) {
                t.setFloat(99f, 2, 3);
                System.out.println("  set [2,3]=99, get [2,3]   = " + t.getFloat(2, 3));
                System.out.println("  untouched [0,0]           = " + t.getFloat(0, 0));
            }

            // 6. ZERO-COPY VIEWS — transpose without data copy
            System.out.println("\nZero-copy transpose");
            try (Tensor mat = Tensor.fill(DType.FLOAT32, 0f, 3, 4)) {
                mat.setFloat(7f, 0, 3); // row 0, col 3
                NDArray transposed = mat.ndarray().transpose();
                System.out.println("  mat[0,3]                  = " + mat.getFloat(0, 3));       // 7.0
                System.out.println("  transposed[3,0]           = " + transposed.getFloat(3, 0)); // 7.0 — same memory
            }

            // 7. toString — formatted print
            System.out.println("\nFormatted output");
            try (Tensor small = Tensor.fill(DType.FLOAT32, 3.14f, 3, 3)) {
                System.out.println(small);
            }
        }

        System.out.println("--- Engine Shutdown Safely ---");
    }
}