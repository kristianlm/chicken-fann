  [FANN]: http://leenissen.dk/fann/wp/
  [Chicken Scheme]: http://call-cc.org/
# [Chicken Scheme] Bindings for [FANN] 

These bindings follow the C API relatively closely, with a few convenience additions.
You can find the original API documentation [here](http://leenissen.dk/fann/html/files/fann-h.html).

Inputs and outputs are lists of numbers. In the future, the API may offer f32vector/f64vectors for 
better performance, but lists have the advantage of being precision-independent.

Returned pointers, such as those from `fann:create-standard` and `fann:read-train-from-list`, currently
need to be freed with their associated destroyers.

## Example 
Here's the [original example](http://leenissen.dk/fann/html/files2/gettingstarted-txt.html) convered to scheme:

```scheme
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
```

Note that data is not fetched from file but from a list with the `fann:read-train-from-list` function. 
Of course, you can still load data from files with `fann:read-train-from-file` as normal.