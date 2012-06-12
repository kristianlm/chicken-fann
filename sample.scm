(use fann)

(define xor-train (fann:read-train-from-list '(((-1 -1) (-1))
                                               ((-1 1) (1))
                                               ((1 -1) (1))
                                               ((1 1) (-1)))))

(define ann (fann:create-standard 2 3 1))

(let* ([max-epochs 500000]
       [epochs-between-reports 1000]
       [desired-error 0.001]
       [num-inputs 2]
       [num-outputs 1])
  
  (fann:set-activation-function-hidden ann fann:sigmoid-symmetric)
  (fann:set-activation-function-output ann fann:sigmoid-symmetric)
  (fann:train-on-data ann xor-train max-epochs epochs-between-reports desired-error))

(define input '(-1 1))
(print input " -> "  (fann:run ann input))

(fann:save ann "/tmp/xor-test.net")
(fann:destroy-train xor-train)
(fann:destroy ann)
