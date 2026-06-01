# FID from Scratch

## Key Insight

[FID](/shared/glossary/#fid) is the standard way to score how realistic a batch of generated images looks, and this project builds it by hand instead of calling a library. You run both real and generated images through a pretrained Inception network to turn each image into a feature vector, summarize each set by its mean and covariance (its center and spread), and plug those into a closed-form distance formula. A lower FID means the two clouds of features overlap more, i.e. the fakes look statistically like the reals. Implementing it yourself demystifies the single number that nearly every image-generation paper reports.
