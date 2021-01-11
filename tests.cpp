#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

template<int D>
float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    // if the combination is 0 0 0, zero should be returned, instead of prod
    bool nonzero = 0;
    for (int i = 0; i < D; i++) {
        nonzero |= combination[i];
        if (combination[i])
            prod *= pow(params[i], ctps[i*2]) * pow(log2(params[i]), ctps[i*2 + 1]);
    }
    return nonzero ? prod : 0;
}

// coefs needs to be D + 1 in size
template<int D>
float evaluate_multi(unsigned char *combination, float *coefs, float *ctps, float *params) {
    float result = coefs[0];
    for (int i = 0; i < D; i++) {
        result += evaluate_single<D>(&combination[i*D], coefs[i + 1], ctps, params);;
    }
    return result;
}

bool check_is_close(float expected, float actual) {
    float atol = 1e-08;
    float rtol = 1e-05;
    if (std::abs(expected - actual) > (atol + rtol * std::abs(actual))) {
        std::cerr << std::setprecision(16) << "Expected " << expected << " got " << actual << std::endl;
        return false;
    }
    return true;
}

void test() {
    // 2D tests
    {
        // Hypothesis: 89.5838036619584 + 4.4425699382361 * p^(1/3) * log2(p)^(1) * q^(1/2) * log2(q)^(2):
        float coefs[] = {89.5838036619584,4.4425699382361};
        float ctps[] = {1./3,1,1./2,2};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {50.0, 4.0};
            float y = 828.5482339289257;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0, 1.0};
            float y = 89.5838036619584;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 109.30422707645451 + 0.0031963099752839656 * p^(11/4) + 0.0005518929195995619 * p^(11/4) * q^(9/4):
        float coefs[] = {109.30422707645451,0.0031963099752839656,0.0005518929195995619};
        float ctps[] = {11./4,0,9./4,0};
        unsigned char combination[] = {1,0,1,1};
        {
            float params[] = {20.0, 3.0};
            float y = 146.12496422556183;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0, 1.0};
            float y = 400.2010849988294;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 134.32303059924104 + 134.32303059924084 *  + 5.124537626161242 *  * q^(5/2) * log2(q)^(1):
        float coefs[] = {134.32303059924104,134.32303059924084,5.124537626161242};
        float ctps[] = {0,0,5./2,1};
        unsigned char combination[] = {1,0,1,1};
        {
            float params[] = {40.0, 3.0};
            float y = 395.25862836954286;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {50.0, 4.0};
            float y = 596.6164692728014;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 132.2688904973669 + 17.58746024664971 * p^(1/4) * q^(4/3) * log2(q)^(1):
        float coefs[] = {132.2688904973669,17.58746024664971};
        float ctps[] = {1./4,0,4./3,1};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {30.0, 5.0};
            float y = 949.4008835563767;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0, 5.0};
            float y = 1104.0080705394953;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 88.78394941196098 + 0.2769972277933893 * p^(1/3) * log2(p)^(2) * q^(9/4):
        float coefs[] = {88.78394941196098,0.2769972277933893};
        float ctps[] = {1./3,2,9./4,0};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {30.0, 5.0};
            float y = 863.503354068158;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0, 3.0};
            float y = 255.13669963919318;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }

    // 3D tests
    {
        // Hypothesis: 118.34952297744546 + 4.710430248637836e-10 * p^(5/2) * q^(1) * log2(q)^(1) * r^(5/2) * log2(r)^(2) + 0.06850282006061759 * p^(5/2) + 1488.2100753462007 * q^(1) * log2(q)^(1):
        float coefs[] = {118.34952297744546,4.710430248637836e-10,0.06850282006061759,1488.2100753462007};
        float ctps[] = {5./2,0,1,1,5./2,2};
        unsigned char combination[] = {1,1,1,1,0,0,0,1,0};
        {
            float params[] = {20.0, 3.0, 400.0};
            float y = 8275.108201722534;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0, 2.0, 400.0};
            float y = 3620.2421732609328;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: -1951.6153507433564 + 7.66811149771554e-08 * p^(1/3) * log2(p)^(2) * q^(1) * r^(3) * log2(r)^(1) + 73.88776806156957 * p^(1/3) * log2(p)^(2):
        float coefs[] = {-1951.6153507433564,7.66811149771554e-08,73.88776806156957};
        float ctps[] = {1./3,2,1,0,3,1};
        unsigned char combination[] = {1,1,1,1,0,0,0,0,0};
        {
            float params[] = {60.0, 2.0, 300.0};
            float y = 12795.390012872915;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {40.0, 1.0, 300.0};
            float y = 6855.619111459854;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 5129.957028551891 + 3.0436541884914803e-06 * p^(9/4) * q^(11/4) * r^(4/3) * log2(r)^(1) + -0.21706486968128272 * p^(9/4) + 0.08302515652013644 * r^(4/3) * log2(r)^(1):
        float coefs[] = {5129.957028551891,3.0436541884914803e-06,-0.21706486968128272,0.08302515652013644};
        float ctps[] = {9./4,0,11./4,0,4./3,1};
        unsigned char combination[] = {1,1,1,1,0,0,0,0,1};
        {
            float params[] = {60.0, 3.0, 100.0};
            float y = 5140.465204920101;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {30.0, 2.0, 300.0};
            float y = 6757.534153538174;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 4085.638897010501 + 0.0001299322067373827 * p^(1/2) * log2(p)^(1) * q^(3) * log2(q)^(1) * r^(1) * log2(r)^(2) + 0.32695395191489945 * p^(1/2) * log2(p)^(1) * q^(3) * log2(q)^(1):
        float coefs[] = {4085.638897010501,0.0001299322067373827,0.32695395191489945};
        float ctps[] = {1./2,1,3,1,1,2};
        unsigned char combination[] = {1,1,1,1,1,0,0,0,0};
        {
            float params[] = {20.0, 5.0, 400.0};
            float y = 27704.080562183488;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0, 3.0, 100.0};
            float y = 4830.460015852253;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: -11827.786873395577 + 1260.0627421055954 * p^(4/5) * q^(1/2) + 0.020041691474716572 * r^(3/2) * log2(r)^(2) + -1199.6827763775277 * p^(4/5):
        float coefs[] = {-11827.786873395577,1260.0627421055954,0.020041691474716572,-1199.6827763775277};
        float ctps[] = {4./5,0,1./2,0,3./2,2};
        unsigned char combination[] = {1,1,0,0,0,1,1,0,0};
        {
            float params[] = {50.0, 5.0, 100.0};
            float y = 26050.63460286099;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {50.0, 3.0, 500.0};
            float y = 28656.57643774028;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
}