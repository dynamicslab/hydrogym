# MSH2NEK

Msh2nek is a Python utility for converting a Fluent mesh (.msh) to a .rea file that can be read by Nek5000. It was initially developed by Mikael Mortensen at University of Oslo. Further modifications were implemented by Jing Gong, Prabal Negi and Ricardo Vinuesa at the Royal Institute of Technology (KTH).

## NOTES
- Supports only hex8 format.
- Curvature can be introduced at the domain boundaries by the user.
- A detailed description of functionalities can be found in the Python file.

## To get started:
For example, starting from a file 'msh_file.msh', with 4 domain boundaries identified by tags 10, 11, 12 and 13, the script can be used as follows:

```
Unix> ipython

...
[1]: from mshconvert import *
[2]: convert('msh_file.msh',bcs={10:'W',11:'v',12:'v',13:'O'},curves={12:{'type': 'm'})
```

Inputs:
- The first argument is the .msh file containing the mesh. 
- The second argument is a Python dictionary. Each key corresponds to one of the boundaries as identified at the end of the msh file. Each value corresponds to the desired boundary condition.
- The third argument is also a dictionary. The value defines the type of curvature.
