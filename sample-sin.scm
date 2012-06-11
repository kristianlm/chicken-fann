(use fann)

(begin
  (define train-list 
    (map (lambda (a)
           `((,a) (,(sin a) )))
         (iota 300 0 0.01)))

  (define train-data (fann:read-train-from-list train-list)))

(begin
  (define ann (fann:create-standard 1 3 1))

  (fann:set-training-algorithm ann fann:train-rprop)
  (fann:set-learning-rate ann 0.7)
  (for-each (cut <> ann fann:sigmoid-symmetric)
            (list fann:set-activation-function-output
                  fann:set-activation-function-hidden))

  (fann:train-on-data ann train-data 20000 500 0.0000008))

(fann:test-data ann train-data)

(pp (map (lambda (I) (cons (fann:run ann I) (sin (car I))))
      '((0)
        (0.5)
        ( 1.6)
        ( 3)) ;;train-input-list
      ))
