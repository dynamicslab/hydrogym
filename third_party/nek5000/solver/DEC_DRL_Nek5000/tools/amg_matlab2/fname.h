#ifndef FNAME_H
#define FNAME_H

/* establishes some macros to establish
   * the FORTRAN naming convention
     default      gs_setup, etc.
     -DUPCASE     GS_SETUP, etc.
     -DUNDERSCORE gs_setup_, etc.
*/

/* the following macro functions like a##b,
   but will expand a and/or b if they are themselves macros */
#define TOKEN_PASTE_(a,b) a##b
#define TOKEN_PASTE(a,b) TOKEN_PASTE_(a,b)

#if defined(UPCASE)
#  define FORTRAN_NAME(low,up) up
#elif defined(UNDERSCORE)
#  define FORTRAN_NAME(low,up) TOKEN_PASTE(low,_)
#else
#  define FORTRAN_NAME(low,up) low
#endif

#endif

