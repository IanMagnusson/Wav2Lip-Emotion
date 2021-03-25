# Lit review


## [A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild]() Prajwal et al 2020
- use an expert descriminator model pretrained on real videos only rather than a descriminator trained in tandem with generator
- propose a new evaluation approach using the pretrained expert model rather than using masked real data which requires model to also
unrealistically have to generate pose changes at test time
