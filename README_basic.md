## EDM Testing

### Data

- Category 1: Non Current based
- Category 2: Current Based

I E D Ra Hm hc
Levels:

- Level 1: 3 inputs, 3 outputs
  - Inputs: I E D
  - Outputs: Ra Hm hc
- Level 2: 6 inputs, 5 outputs
  - Inputs: I E D Ra Hm hc
  - Outputs: Ff COF Wd T WL
- Level 3: 11 inputs, 1 outputs
  - Inputs: I E D Ra Hm hc Ff COF Wd T Wl
  - Outputs: k Keff

### Data Mapping :

> Note: please provide units /scales for each input/output

#### Level-1:

1. I:
   - Discharge Current (Values, float valeus (4.5 - 15))
2. E:
   - Electrode - Category: G(Graphite) or C (copper)
3. D:
   - Dielectric - O(Parafin oil), or W(Distilled Water)
4. RA: Surface Roughness
5. HM: Hardness (Either in scale 1 or scale 2)
6. hc: Recaste Layer thickness

#### Level 2:

- HM: Hardness (scale 2)

7. FF: Fraction Force
8. COF: Coeff of Fraction
9. Wd: Wear Depth
10. T: Temperature
11. WL: Weight Loss

#### Level 3:

12. k: Orchid Wear Coeffecient
13. K-Eff: Effective farcture resistance
