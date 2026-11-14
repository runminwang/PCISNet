# PCISNet
Robust Scene Text Recognition via Cross-View Posterior Consistency and Intermediate Sequence Supervision

Local corruption and occlusion remain challenging for scene text recognition based on connectionist temporal classification (CTC). We propose PCISNet, a two-phase training framework that combines cross-view posterior consistency with intermediate sequence supervision. The first phase constructs two views on a shared spatial grid by independently replacing full-height vertical strips with Gaussian noise. A shared recognizer processes both views under CTC supervision, while symmetric Kullback–Leibler divergence encourages agreement between their position-wise posteriors. The second phase initializes the encoder with the learned parameters and switches to main and auxiliary CTC objectives, providing direct sequence supervision at an intermediate visual stage. PCISNet achieves competitive or superior performance compared to existing state-of-the-art methods on several English and Chinese benchmarks, improving SVTRv2-B’s average word accuracy by 1.69 and 4.57 percentage points on Union14M-Benchmark and OST, respectively. Extensive experiments with three additional recognizers further support its applicability across CTC architectures. 

-----------------------------------------November 14, 2026-----------------------------------------

This work has been submitted to Neurocomputing. After the paper is accepted, we will release all code as soon as possible.


