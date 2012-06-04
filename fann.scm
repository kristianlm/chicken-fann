
(module fann *
 (import chicken scheme foreign srfi-4 bind)
#>
#include <fann.h>
<#

(bind-options default-renaming: "fann:")
(bind-rename/pattern "^fann-" "")
(bind-include-path "./include")
(bind-file "include/fann.h")


 )
