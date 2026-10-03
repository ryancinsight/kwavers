<a id="kw-book-ch29-residuals"></a>

## KW-BOOK-CH29-RESIDUALS — Chapter 29 figures 5 and 6 cannot be regenerated [patch] — todo

priority: verification; needs: none; scope: `docs/book` chapter 29 figures and their plotting code; the nonlinear brain PyO3 binding

- **Finding.** Figure 5's regeneration is blocked by the nonlinear brain PyO3 allocation abort, so the checked-in pressure column came from the controlled CT-frame field archive; the same block holds figure 6.
- **Impact.** Two book figures cannot be reproduced from committed plotting code, the property the book's figures are supposed to have.
- **Acceptance:** both figures regenerate from committed code, or the chapter states which archive they came from and why.
- **Next step:** reproduce the allocation abort in the nonlinear brain binding; fix it or record the archive provenance in the chapter.
