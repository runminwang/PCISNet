# PCISNet
Posterior Consistency and Intermediate Supervision Network for Robust Scene Text Recognition

Connectionist temporal classification (CTC) supports efficient scene text recognition, but local visual corruption can destabilize its character posteriors. Moreover, supervision applied only at the final encoder output may not sufficiently constrain intermediate sequence representations. We address these limitations with PCISNet, a training framework that combines posterior consistency and intermediate supervision in two sequential phases. The first phase replaces full-height vertical strips with Gaussian noise in two independently perturbed views and regularizes their character posteriors using symmetric Kullback--Leibler divergence. The second phase initializes the encoder from the first-phase checkpoint and applies an auxiliary CTC loss at an intermediate encoder stage. Compared with the baseline, PCISNet improves average word accuracy by 0.20, 1.69, and 4.57 percentage points on the common English benchmarks, Union14M-Benchmark, and the Occlusion Scene Text benchmark, respectively. The corresponding gain on the Benchmarking Chinese Text Recognition (BCTR) benchmark is 0.68 percentage points. The auxiliary branch is discarded after training,  preserving the original parameter count and inference path. These results show that targeted training supervision can improve CTC robustness without complicating deployment.

-----------------------------------------October 25, 2026-----------------------------------------

This work has been submitted to IEEE Transactions on Circuits and Systems for Video Technology. After the paper is accepted, we will release all code as soon as possible.


