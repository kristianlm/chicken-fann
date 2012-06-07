
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

;; convert a (c-pointer fann_type) to a list (only f32vector for now)
(define (pointer->list pointer len)
  (fvector->list
   (blob->fvector/shared
    (pointer->blob pointer
                   (fx* len (foreign-value "sizeof(fann_type)" int))))))


(define (fann:create-standard . layer-sizes)
 (fann:create-standard-array (length layer-sizes)
                             (list->u32vector layer-sizes)))

;; some convenience conversions for arguments and return
(let-syntax ([redefine (lambda (x r t)
                         (let ([func (caadr x)]
                               [arglist (cdadr x)])
                           `(set! ,func
                              (let ([$ ,func])
                                (lambda ,arglist
                                  ,@(cddr x))))))])

  (redefine (fann:run ann inputs)
            (pointer->list ($ ann (list->fvector inputs))
                           (fann:get-num-output ann)))
  
  (redefine (fann:test ann inputs outputs)
            (pointer->list ($ ann (list->fvector inputs) (list->fvector outputs))
                           (fann:get-num-output ann)))

  (redefine (fann:train ann inputs outputs)
            ($ ann (list->fvector inputs) (list->fvector outputs))))


;;; Allow creating trainging-sets from lists
(define *data-tuples* #f)

(define-external (train_from_list_callback (unsigned-integer idx)
                              (unsigned-integer num_inputs)
                              (unsigned-integer num_outputs)
                              ((c-pointer "fann_type") vinput)
                              ((c-pointer "fann_type") voutput))
  void

  (define in-list (car (list-ref *data-tuples* idx)))
  (define out-list (cadr (list-ref *data-tuples* idx)))

  (move-memory! (f32vector->blob/shared (list->f32vector in-list))
                vinput
                (fx* num_inputs (foreign-value "sizeof(fann_type)" int)))
  (move-memory! (f32vector->blob/shared (list->f32vector out-list))
                voutput
                (fx* num_outputs (foreign-value "sizeof(fann_type)" int))))


(define (fann:read-train-from-list data-tuples)
  (let ([in-len (length (caar data-tuples))]
        [out-len (length (cadar data-tuples))])
    ;; hacky but seems to work. we can't supply tuples in parameter
    ;; because method signature is fixed by callback typedef
    (set! *data-tuples* data-tuples)
    (let ([train-struct-pointer
           ((foreign-safe-lambda* (c-pointer "fann_train_data")
                             ((unsigned-integer num_data)
                              (unsigned-integer num_inputs)
                              (unsigned-integer num_outputs))
                             "C_return(fann_create_train_from_callback(num_data, num_inputs, num_outputs, &train_from_list_callback));")
            (length data-tuples) in-len out-len)])
      ;; release data-holder! (may be a lot of data!)
      (set! *data-tuples* #f)
      train-struct-pointer)))


)
