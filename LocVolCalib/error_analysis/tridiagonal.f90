! Solve the tridiagonal system with DGTRFSX and print 
! error bounds and relative error compared to the naive solution.
! Assumes low.bin, diag.bin, high.bin, res.bin to have been generated by
! PrintImplicitX.cpp

program main
    integer :: io_low, io_diag, io_high, io_b, io_res
    real(kind=8), allocatable :: low(:), diag(:), high(:), b(:,:), res(:, :)
    integer, allocatable :: ipiv(:), iwork(:)
    real(kind=8), allocatable :: dlf(:), df(:), duf(:), du2(:), &
                                 ferr(:), berr(:), work(:), x(:, :)
    real(kind=8)  :: rcond
    real(kind=8), allocatable :: stat(:, :)
    integer :: n, info

    ! numX
    n = 32 ! small
!    n = 256 ! large

    allocate(low(n - 1), diag(n), high(n - 1), b(n, 1), res(n, 1))
    allocate(dlf(n - 1), df(n), duf(n), du2(n - 2), ipiv(n), x(n, 1), &
             ferr(n), berr(n), work(3 * n), iwork(n))
    allocate(stat(n, 3))
    
    open(newunit=io_low, file="low.bin", form="unformatted", status="old", &
                action="read", access="stream")
    read(io_low) low
    close(io_low)
    open(newunit=io_diag, file="diag.bin", form="unformatted", status="old", &
                action="read", access="stream")
    read(io_diag) diag
    close(io_diag)
    open(newunit=io_high, file="high.bin", form="unformatted", status="old", &
                action="read", access="stream")
    read(io_high) high
    close(io_high)
    open(newunit=io_b, file="b.bin", form="unformatted", status="old", &
                action="read", access="stream")
    read(io_b) b
    close(io_b)
    open(newunit=io_res, file="res.bin", form="unformatted", status="old", &
                action="read", access="stream")
    read(io_res) res
    close(io_res)

    call DGTSVX('N', 'N', n, 1, low, diag, high, dlf, df, duf, du2, ipiv, &
                  b, n, x, n, rcond, ferr, berr, work, iwork, info)

    if (info .eq. 0) then
        write(*, *) "Estimated condition number: ", 1.0 / rcond
        write(*, *) "Forward error bound lapack: ", maxval(ferr(:))
        stat(:, 1) = res(:, 1)
        stat(:, 2) = x(:, 1)
        stat(:, 3) = abs(res(:, 1) - x(:, 1)) / abs(x(:, 1))
        write(*, *) "C++, lapack, relative error"
        do i = 1, n
            write(*, *) stat(i,:)
        end do
    else
        write(*, *) "Error: ", info
    endif
                

    deallocate(low, diag, high, b, res)
    deallocate(dlf, df, duf, du2, ipiv, x, ferr, berr, work, iwork)
    deallocate(stat)
end program