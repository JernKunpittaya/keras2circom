pragma circom 2.0.0;

template MeanCheck (nInputs) {
    signal input in[nInputs];
    signal input out[1];
    
    signal sum_till[nInputs];
    sum_till[0] <== in[0];
    for (var i = 1; i<nInputs; i++){
        sum_till[i] <== sum_till[i-1]+in[i];
    }
    sum_till[n-1] === out[0]*nInputs;
}