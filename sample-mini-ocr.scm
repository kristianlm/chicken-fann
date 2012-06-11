(use fann octave)
(octave:start)

;; eg 4 -> '(0 0 0 0 1 0 0 0 0 0)
(define (digit->vector digit)
  (map (lambda (idx) (if (= digit idx) 1 0)) (iota 10)))

(define (drawing->input symdrawing) 
  (let ([char->num (lambda (char)
                     (case char
                       ([#\.] 0)
                       ([#\o] 1)) )])
    (flatten
     (map (compose (lambda (charlist)
                     (map char->num charlist))
                   string->list symbol->string)
          symdrawing))))

(begin
  (define raw-digits
    '(((..oo..
        oo..o.
        ...o..
        ..o...
        .oooo.) 2)


      ((.ooo..
        o...o.
        .oooo.
        ....o.
        ....o.) 9)

      ((..oo..
        .o..o.
        ..ooo.
        ....o.
        ....o.) 9)

      ((.ooo..
        o..o..
        .ooo..
        ...o..
        ...o..) 9)

      ((.oo...
        o..o..
        .ooo..
        ...o..
        ..o...) 9)
            
      ((
        ......
        .o..o.
        .o.oo.
        ....o.
        ...o..) 4)

      ((..oo..
        ....o.
        ...oo.
        ..o...
        ..ooo.) 2)


      ((......
        oo..o.
        ...oo.
        ..o...
        .oooo.) 2)

    
      ((....o.
        ....o.
        ....o.
        ....o.
        ......) 1)

      ((......
        ...oo.
        ....o.
        ....o.
        ......) 1)

      ((o.....
        o.....
        o.....
        o.....
        ......) 1)

      ((...ooo
        .oo..o
        .....o
        .....o
        .....o) 7)

      ((......
        ..o...
        ..o...
        ..o...
        ......) 1)

      ((oooooo
        .....o
        ...ooo
        ....o.
        ....o.) 7)
    
      ((..oooo
        .....o
        .....o
        .....o
        .....o) 7)


      ((..oo..
        .o..o.
        ..oo..
        .o..o.
        ..oo..) 8)

      ((..oo..
        .o..o.
        ..oo..
        ..o..o
        ..ooo.) 8)

      ((.o..o.
        oo..o.
        .oooo.
        ....o.
        ....o.) 4)

      ((......
        ..o.o.
        ..ooo.
        ....o.
        ....o.) 4)
    
      ((....o.
        o...o.
        ooooo.
        ....o.
        ....o.) 4)

      ((.ooo..
        o...o.
        .oooo.
        .o..o.
        ..oo..) 8)))

  
  (define data-list
    (map (lambda (tuple-num)
           (let ([sdrawing (car tuple-num)])
             (list (drawing->input sdrawing) (digit->vector (cadr tuple-num)))))
         raw-digits))


  (define train (fann:read-train-from-list data-list)))


(begin
  (define ann (fann:create-standard 30 25 10))
;  (fann:set-training-algorithm ann fann:train-quickprop)
  (fann:set-activation-function-hidden ann fann:sigmoid)
  (fann:set-activation-function-output ann fann:sigmoid)
  
  (fann:train-on-data ann train 10000 1000 0.000001))

(fann:test-data ann train)
(fann:num-input-train-data train)


(begin
  (octave:bar (fann:run ann (drawing->input
                             '(..oo..
                               .o..o.
                               ...o..
                               ..o...
                               ...oo.))))
  (octave:send "refresh;\n"))
