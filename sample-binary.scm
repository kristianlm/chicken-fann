(use fann)

(define train-list '(((0 0) (0))
                     ((0 1) (1))
                     ((1 0) (2))
                     ((1 1) (3))))

(define train-input-list (map (cut car <>)
                              train-list))

(define train-data (fann:read-train-from-list train-list))

(define ann (fann:create-standard 2 3 1))

(for-each (cut <> ann fann:linear)
          (list fann:set-activation-function-output
                fann:set-activation-function-hidden))

(fann:train-on-data ann train-data 1000 1 0.001)
(fann:test-data ann train-data)

(map (lambda (I) (fann:run ann I))
     train-input-list)
