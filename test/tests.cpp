#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

template<int D>
float evaluate_single(unsigned char *combination, float coef, float *ctps, float *params) {
    float prod = coef;
    // if the combination is 0 0 0, zero should be returned, instead of prod
    bool nonzero = false;
    for (int i = 0; i < D; i++) {
        nonzero |= (bool) combination[i];
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
    float atol = 1e-08f;
    float rtol = 1e-05f;
    if (std::abs(expected - actual) > (atol + rtol * std::abs(actual))) {
        std::cerr << std::setprecision(16) << "Expected " << expected << " got " << actual << std::endl;
        return false;
    }
    return true;
}

void test() {
    // 2D testdata
    {
        // Hypothesis: 89.5838036619584 + 4.4425699382361 * p^(1/3) * log2(p)^(1) * q^(1/2) * log2(q)^(2):
        float coefs[] = {89.5838036619584f, 4.4425699382361f};
        float ctps[] = {1./3,1,1./2,2};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {50.0f, 4.0f};
            float y = 828.5482339289257f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0f, 1.0f};
            float y = 89.5838036619584f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 109.30422707645451 + 0.0031963099752839656 * p^(11/4) + 0.0005518929195995619 * p^(11/4) * q^(9/4):
        float coefs[] = {109.30422707645451f, 0.0031963099752839656f, 0.0005518929195995619f};
        float ctps[] = {11.f/4, 0.f, 9.f/4, 0.f};
        unsigned char combination[] = {1,0,1,1};
        {
            float params[] = {20.0f, 3.0f};
            float y = 146.12496422556183f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0f, 1.0f};
            float y = 400.2010849988294f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 134.32303059924104 + 134.32303059924084 *  + 5.124537626161242 *  * q^(5/2) * log2(q)^(1):
        float coefs[] = {134.32303059924104f,134.32303059924084f,5.124537626161242f};
        float ctps[] = {0.f,0.f,5.f/2,1.f};
        unsigned char combination[] = {1,0,1,1};
        {
            float params[] = {40.0f, 3.0f};
            float y = 395.25862836954286f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {50.0f, 4.0f};
            float y = 596.6164692728014f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 132.2688904973669 + 17.58746024664971 * p^(1/4) * q^(4/3) * log2(q)^(1):
        float coefs[] = {132.2688904973669f,17.58746024664971f};
        float ctps[] = {1.f/4,0.f,4.f/3,1.f};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {30.0f, 5.0f};
            float y = 949.4008835563767f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {60.0f, 5.0f};
            float y = 1104.0080705394953f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 88.78394941196098 + 0.2769972277933893 * p^(1/3) * log2(p)^(2) * q^(9/4):
        float coefs[] = {88.78394941196098f,0.2769972277933893f};
        float ctps[] = {1.f/3,2.f,9.f/4,0.f};
        unsigned char combination[] = {1,1,0,0};
        {
            float params[] = {30.0f, 5.0f};
            float y = 863.503354068158f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0f, 3.0f};
            float y = 255.13669963919318f;
            float yp = evaluate_multi<2>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }

    // 3D testdata
    {
        // Hypothesis: 118.34952297744546 + 4.710430248637836e-10 * p^(5/2) * q^(1) * log2(q)^(1) * r^(5/2) * log2(r)^(2) + 0.06850282006061759 * p^(5/2) + 1488.2100753462007 * q^(1) * log2(q)^(1):
        float coefs[] = {118.34952297744546f,4.710430248637836e-10f,0.06850282006061759f,1488.2100753462007f};
        float ctps[] = {5.f/2,0.f,1.f,1.f,5.f/2,2.f};
        unsigned char combination[] = {1,1,1,1,0,0,0,1,0};
        {
            float params[] = {20.0f, 3.0f, 400.0f};
            float y = 8275.108201722534f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0f, 2.0f, 400.0f};
            float y = 3620.2421732609328f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: -1951.6153507433564 + 7.66811149771554e-08 * p^(1/3) * log2(p)^(2) * q^(1) * r^(3) * log2(r)^(1) + 73.88776806156957 * p^(1/3) * log2(p)^(2):
        float coefs[] = {-1951.6153507433564f,7.66811149771554e-08f,73.88776806156957f};
        float ctps[] = {1.f/3,2.f,1.f,0.f,3.f,1.f};
        unsigned char combination[] = {1,1,1,1,0,0,0,0,0};
        {
            float params[] = {60.0f, 2.0f, 300.0f};
            float y = 12795.390012872915f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {40.0f, 1.0f, 300.0f};
            float y = 6855.619111459854f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 5129.957028551891 + 3.0436541884914803e-06 * p^(9/4) * q^(11/4) * r^(4/3) * log2(r)^(1) + -0.21706486968128272 * p^(9/4) + 0.08302515652013644 * r^(4/3) * log2(r)^(1):
        float coefs[] = {5129.957028551891f,3.0436541884914803e-06f,-0.21706486968128272f,0.08302515652013644f};
        float ctps[] = {9.f/4,0.f,11.f/4,0.f,4.f/3,1.f};
        unsigned char combination[] = {1,1,1,1,0,0,0,0,1};
        {
            float params[] = {60.0f, 3.0f, 100.0f};
            float y = 5140.465204920101f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {30.0f, 2.0f, 300.0f};
            float y = 6757.534153538174f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: 4085.638897010501 + 0.0001299322067373827 * p^(1/2) * log2(p)^(1) * q^(3) * log2(q)^(1) * r^(1) * log2(r)^(2) + 0.32695395191489945 * p^(1/2) * log2(p)^(1) * q^(3) * log2(q)^(1):
        float coefs[] = {4085.638897010501f,0.0001299322067373827f,0.32695395191489945f};
        float ctps[] = {1.f/2,1.f,3.f,1.f,1.f,2.f};
        unsigned char combination[] = {1,1,1,1,1,0,0,0,0};
        {
            float params[] = {20.0f, 5.0f, 400.0f};
            float y = 27704.080562183488f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {20.0f, 3.0f, 100.0f};
            float y = 4830.460015852253f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
    {
        // Hypothesis: -11827.786873395577 + 1260.0627421055954 * p^(4/5) * q^(1/2) + 0.020041691474716572 * r^(3/2) * log2(r)^(2) + -1199.6827763775277 * p^(4/5):
        float coefs[] = {-11827.786873395577f,1260.0627421055954f,0.020041691474716572f,-1199.6827763775277f};
        float ctps[] = {4.f/5,0.f,1.f/2,0.f,3.f/2,2.f};
        unsigned char combination[] = {1,1,0,0,0,1,1,0,0};
        {
            float params[] = {50.0f, 5.0f, 100.0f};
            float y = 26050.63460286099f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
        {
            float params[] = {50.0f, 3.0f, 500.0f};
            float y = 28656.57643774028f;
            float yp = evaluate_multi<3>(combination, coefs, ctps, params);
            check_is_close(y, yp);
        }
    }
}