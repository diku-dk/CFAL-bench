module nbody_mod
contains
    subroutine accelerateAll(accel, positions, masses, n)
        implicit none
    
        integer, intent(in) :: n
        real(KIND=8), intent(in) :: masses(n), positions(n, 3)
        real(KIND=8), intent(out) :: accel(n, 3)
        integer :: i, j
        real(KIND=8) :: a(3), diff(3), norm, factor

        do i = 1, n
            a = 0 ! Have to lift manually in order for GCC to vectorize
            do j = 1, n
                diff = positions(j, :) - positions(i, :)
                norm = sqrt(sum(diff * diff)) ** 3

!                factor = merge(0.0, masses(j) / norm, norm .eq. 0.0)
                factor = merge(0.0D0, masses(j) / norm, norm .eq. 0.0)

                a = a + diff * factor
            end do
            accel(i,:) = a
        end do
    end subroutine accelerateAll
    
    subroutine advanceIt(positions, velocities, masses, accel, dt, n)
        implicit none
    
        integer, intent(in) :: n
        real(KIND=8), intent(in) :: dt, masses(n)
        real(KIND=8), intent(inout) :: positions(n, 3), velocities(n, 3), &
                                       accel(n, 3)
    
        call accelerateAll(accel, positions, masses, n)
        velocities = velocities + accel * dt
        positions = positions + velocities * dt
    end subroutine advanceIt
end module nbody_mod

program nbody
    use nbody_mod
    implicit none

    integer :: n, iterations, t, i
    real(KIND=8), allocatable :: positions(:, :), velocities(:, :), & 
                                 accel(:, :), masses(:)
    real(KIND=8) :: dt
    integer :: cpu_count, cpu_count2, count_rate, count_max
    character(len=12), dimension(:), allocatable :: args

    allocate(args(2))
    call get_command_argument(1, args(1))
    read (unit=args(1), fmt=*) n

    call get_command_argument(2, args(2))
    read (unit=args(2), fmt=*) iterations

    allocate(positions(n, 3))
    allocate(velocities(n, 3))
    allocate(accel(n, 3))
    allocate(masses(n))
 
    do i = 1, n
        positions(i, 1) =      i - 1
        positions(i, 2) = 2 * (i - 1)
        positions(i, 3) = 3 * (i - 1)
    end do
    velocities = 0
    masses = 1

    dt = 0.01

    call system_clock(cpu_count, count_rate, count_max)
    do t = 1, iterations
        call advanceIt(positions, velocities, masses, accel, dt, n)
    end do
    call system_clock(cpu_count2, count_rate, count_max)

    do i = 1, n
        write (0, *) positions(i,:)
    end do

    print *, (18.0 * n * n + 12.0 * n) * iterations / 1E9 / &
                    (real(cpu_count2 - cpu_count) / count_rate)

    deallocate(positions)
    deallocate(velocities)
    deallocate(accel)
    deallocate(masses)

end program nbody
