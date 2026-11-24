//! Pairs of neighbor indices for periodic and endpoint-reflecting grid axes.

#[inline]
pub(super) fn periodic(i: usize, d: usize, len: usize) -> (usize, usize) {
    let step = if d < len { d } else { d % len };
    let minus = if i >= step {
        i - step
    } else {
        len - (step - i)
    };
    let plus = if step < len - i {
        i + step
    } else {
        step - (len - i)
    };
    (minus, plus)
}

#[inline]
pub(super) fn reflect(i: usize, d: usize, len: usize) -> (usize, usize) {
    if len == 1 {
        return (0, 0);
    }
    let last = len - 1;
    let period = 2 * last;
    let fold = |j| last - last.abs_diff(j % period);
    (fold(i.abs_diff(d)), fold(i + d))
}
