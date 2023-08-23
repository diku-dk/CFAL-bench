module fft_mod
contains
    ! Simple recursive algorithm, assumes length of x is power of two
    recursive subroutine fft(x)
        complex, intent(inout) :: x(:)
        complex :: temp
        integer :: n, k
        real :: pi

        n = size(x)
        pi = acos(-1.0)

        if (n .eq. 1) then
            return
        endif

        call fft(x(1:n:2))
        call fft(x(2:n:2))

        do k = 1, n, 2
            temp = x(k + 1) * exp(-2.0 * pi * complex(0, 1) / n * (k / 2))
            x(k + 1) = x(k) - temp
            x(k) = x(k) + temp
        end do
    end subroutine

    subroutine fft3d(x)
        complex, intent(inout) :: x(:,:,:)
        integer :: n1, n2, n3

        n1 = size(x, 1)
        n2 = size(x, 2)
        n3 = size(x, 3)

        do j = 1, n2
            do k = 1, n3
                call fft(x(:,j,k))
            end do
        end do

        do i = 1, n1
            do k = 1, n3
                call fft(x(i,:,k))
            end do
        end do

        do i = 1, n1
            do j = 1, n2
                call fft(x(i,j,:))
            end do
        end do
    end subroutine
end module

program main
    use fft_mod
    implicit none
    complex, allocatable :: x(:,:,:)
    character(len=12) :: arg
    integer :: n

    call get_command_argument(1, arg)
    read (unit=arg, fmt=*) n
    allocate(x(n,n,n))

    x(:,:,:) = 0
    x(1,1,1) = complex(1, 0)

    call fft3D(x)

    write(*, *) sum(x)
    write(*, *) "Should be ", n * n * n

    deallocate(x)

end program main
