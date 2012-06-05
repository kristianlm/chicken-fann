
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

(declare (hide pointer->blob
               fvector->list
               list->fvector
               blob->fvector/shared
               pointer->list))

;; make life easier if we want to change precision
(define fvector->list f32vector->list)
(define list->fvector list->f32vector)
(define blob->fvector/shared blob->f32vector/shared)

(define (pointer->blob pointer bytes)
  (let ([b (make-blob bytes)])
    (move-memory! pointer b bytes)
    b))


(define (pointer->list ann pointer)
  (fvector->list
   (blob->fvector/shared
    (pointer->blob pointer
                   (fx* (foreign-value "sizeof(fann_type)" int)
                        (fann:get-num-output ann))))))


(define (fann:create-standard . layer-sizes)
 (fann:create-standard-array (length layer-sizes)
                             (location
                              (u32vector->blob
                               (list->u32vector layer-sizes)))))

;; some convenience conversions for arguments and return
(let-syntax ([redefine (lambda (x r t)
                         (let ([func (caadr x)]
                               [arglist (cdadr x)])
                           `(set! ,func
                              (let ([$ ,func])
                                (lambda ,arglist
                                  ,@(cddr x))))))])

  (redefine (fann:run ann inputs)
            (pointer->list ann ($ ann (list->fvector inputs))))
  
  (redefine (fann:test ann inputs outputs)
            (pointer->list ann ($ ann (list->fvector inputs) (list->fvector outputs))))

  (redefine (fann:train ann inputs outputs)
            ($ ann (list->fvector inputs) (list->fvector outputs))))



)
