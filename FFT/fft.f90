! TODO finish this implementation

subroutine FFT(x, weights, n)
    integer, intent(in) :: n
    complex, intent(inout) :: x(n), weights(n / 2)
    complex :: temp

    if (n .eq. 1) then
        return
    endif

    FFT(x(1:n:2), n / 2) ! Transform even part
    FFT(x(2:n:2), n / 2) ! Transform odd part

    x_even = x(0:n:2)
    x_odd = x(1:n:2) * weights

    x(1:n/2) = x_even + x_odd
    x(n/2 + 1:n) = x_even - x_odd
end function
