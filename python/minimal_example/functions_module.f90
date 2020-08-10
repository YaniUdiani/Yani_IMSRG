module functions_module
    implicit none

    contains
    function timesthree_fortran( input ) result( output ) bind(c)
        integer, value, intent(in) :: input
        integer :: output
        output = 3*input
        return
    end function

end module
