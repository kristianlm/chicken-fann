
(module fann *
 (import chicken scheme foreign)
 (import bind)
 (use srfi-4 lolevel)

#>
#include <fann.h>
<#

(bind-options default-renaming: "fann:" export-constants: #t)
(bind-rename/pattern "^fann-" "")
(bind-include-path "./include")
(bind-file "include/fann.h")

(define fann:sizeof-uint (foreign-value "sizeof(unsigned int)" int))
(define fann:sizeof-fann-type (foreign-value "sizeof(fann_type)" int))

(define (fann:create-standard . layer-sizes)
 (fann:create-standard-array (length layer-sizes)
                             (location
                              (u32vector->blob
                               (list->u32vector layer-sizes)))))

(define (pointer->blob pointer bytes)
  (let ([b (make-blob bytes)])
    (move-memory! pointer b bytes)
    b))

(define fann:run
  (let ([fann:run* fann:run])
    (lambda (ann inputs)
      (f32vector->list
       (blob->f32vector/shared
        (pointer->blob (fann:run* ann (list->f32vector inputs))
                       (fx* (foreign-value "sizeof(fann_type)" int)
                            (fann:get-num-output ann))))))))



 )
