#ifdef WIN32
#   include <windows.h>
#endif
#ifndef APIENTRY
#   if defined(__CYGWIN__)
#       define APIENTRY __attribute__ ((__stdcall__))
#   else
#       define APIENTRY
#  endif
#endif

#ifdef __APPLE__
#   include <OpenGL/gl.h>
#else
#   include <GL/gl.h>
#endif  // __APPLE__