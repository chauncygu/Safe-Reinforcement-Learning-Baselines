/** To this day I have no idea how to write a makefile.

The platform-independent way of including the julia-to-c library is to pull the compile flags from this julia-config.jl script.

 Run as
    test:
        echo "\n\n"
        /opt/julia-1.8.0/share/julia/julia-config.jl --cflags --ldflags --ldlibs | xargs gcc postshield.c -o postshield.o -D TEST=true
        ./postshield.o

    library:
        echo "\n\n"
        /opt/julia-1.8.0/share/julia/julia-config.jl --cflags --ldflags --ldlibs | xargs gcc -c -fPIC  postshield.c -o postshield.o
        /opt/julia-1.8.0/share/julia/julia-config.jl --cflags --ldflags --ldlibs | xargs gcc -shared -o libccpostshield.so postshield.o


*/

#include <stdio.h>
#include <stdbool.h>
#include <julia.h>


#define JULIA(str) \
    jl_eval_string(str);\
    if (jl_exception_occurred())\
        fprintf(stderr, "Julia Exception:\t%s \nWhile running command:\t%s\n", jl_typeof_str(jl_exception_occurred()), str);

#define GENEROUS_STRING_LENGTH  511


// Consts for testing only
#define CODE_PATH "/path/to/ReproducibilityPackage"
# define POSTSHIELD_NOTEBOOK_PATH CODE_PATH "fig-CCShieldingResultsGroup/PostShield Strategy.jl"
#define STRATEGY_PATH CODE_PATH "/Export/CC-shielded.json"
#define SHIELD_PATH CODE_PATH "/Export/CCShields/old testshield CC 192 samples with G of 0.5.shield"
#define PKG_PROJECT_PATH CODE_PATH

bool already_running = false;

int my_concat3(char out[], int maxsize, const char s1[], const char s2[], const char s3[]) 
{
    
    int result_length = snprintf(out, maxsize,
        "%s%s%s", s1, s2, s3);

    if (result_length > maxsize)
        fprintf(stderr, "format string overflow: %d\n", result_length);

    return result_length > maxsize;
}

// Spin up jl runtime and load functions shielded_strategy and intervened
int initialize_julia_context(const char postshield_notebook_path[], const char strategy_path[], const char shield_path[], const char pkg_project_path[])
{
    if (already_running){
        return 0;
    }

    jl_init();
    
    JULIA("using Pkg");

    const int L = GENEROUS_STRING_LENGTH;
    char statement1[L], statement2[L], statement3[L], statement4[L];

    my_concat3(statement1, L, "Pkg.activate(\"", pkg_project_path, "\", io=devnull)");
    my_concat3(statement2, L, "include(\"", postshield_notebook_path, "\")");
    my_concat3(statement3, L, "strategy_path = \"", strategy_path, "\"");
    my_concat3(statement4, L, "shield_path = \"", shield_path, "\"");

    //JULIA("Activate .");  //TODO: This should be uncommented upon migration to ReproducibilityPackage.
    JULIA(statement1);
    JULIA(statement2);
    JULIA(statement3);
    JULIA(statement4);

    // In cases where the shield intervenes, the allowed action is chosen by the policy from remaining safe actions.
    JULIA("deterministic_shielded_strategy = get_shielded_strategy_int(strategy_path, shield_path, true)");
    
    // In cases where the shield intervenes, all safe actions are allowed.
    JULIA("nondeterministic_shielded_strategy = get_shielded_strategy_int(strategy_path, shield_path, false)");
    JULIA("intervened = get_intervention_checker(strategy_path, shield_path)");


    if (jl_exception_occurred())
    {
        fprintf(stderr, "Error initializing julia context.\n");
        return 1;
    }

    already_running = true;
    return 0;
}

bool intervened(double v_ego, double v_front, double distance)
{
    char intervened_call[GENEROUS_STRING_LENGTH];

    int result_length = snprintf(intervened_call, sizeof(intervened_call),
        "intervened((%f, %f, %f))", v_ego, v_front, distance);

    if (result_length > GENEROUS_STRING_LENGTH)
        fprintf(stderr, "format string overflow\n");


    jl_value_t *boxed_result = JULIA(intervened_call);

    if (jl_typeis(boxed_result, jl_bool_type))
    {
        bool result = jl_unbox_bool(boxed_result);
        return result;
    }
    else
    {
        fprintf(stderr, "unexpected return type\n");
    }
}

// In cases where the shield intervenes, all safe actions are returned.
int nondeterministic_shielded_strategy(double v_ego, double v_front, double distance)
{
    if (!already_running)
    {
        fprintf(stderr, "Operation failed because julia context is not running.\n");
        return -1;
    }

    char intervened_call[GENEROUS_STRING_LENGTH];

    int result_length = snprintf(intervened_call, sizeof(intervened_call),
        "nondeterministic_shielded_strategy((%f, %f, %f))", v_ego, v_front, distance);

    if (result_length > GENEROUS_STRING_LENGTH)
        fprintf(stderr, "format string overflow\n");


    jl_value_t *boxed_result = JULIA(intervened_call);

    if (jl_typeis(boxed_result, jl_int64_type))
    {
        int result = jl_unbox_int64(boxed_result);
        return result;
    }
    else
    {
        fprintf(stderr, "unexpected return type\n");
        return -1;
    }
}

// In cases where the shield intervenes, the policy chooses the corrected action, and this is what is returned.
int deterministic_shielded_strategy(double v_ego, double v_front, double distance)
{
    if (!already_running)
    {
        fprintf(stderr, "Operation failed because julia context is not running.\n");
        return -1;
    }
    
    char intervened_call[GENEROUS_STRING_LENGTH];

    int result_length = snprintf(intervened_call, sizeof(intervened_call),
        "deterministic_shielded_strategy((%f, %f, %f))", v_ego, v_front, distance);

    if (result_length > GENEROUS_STRING_LENGTH)
        fprintf(stderr, "format string overflow\n");


    jl_value_t *boxed_result = JULIA(intervened_call);

    if (jl_typeis(boxed_result, jl_int64_type))
    {
        int result = jl_unbox_int64(boxed_result);
        return result;
    }
    else
    {
        fprintf(stderr, "unexpected return type\n");
        return -1;
    }
}


int main()
{
    /*
    foo("baz");

    char concat_result[GENEROUS_STRING_LENGTH];
    my_concat3(concat_result, GENEROUS_STRING_LENGTH, "Testing my_concat:", " It works.", " - yay!\n");
    printf("%s", concat_result);
    /**/
    
    initialize_julia_context(POSTSHIELD_NOTEBOOK_PATH, STRATEGY_PATH, SHIELD_PATH, PKG_PROJECT_PATH);

    printf("intervened((3, 3, 37)):\t%d\t(should be: 1)\n", intervened(3, 3, 37));
    printf("intervened((3, 3, 41)):\t%d\t(should be: 1)\n", intervened(3, 3, 41));
    printf("intervened((10, 3, 81)):\t%d\t(should be: 1)\n", intervened(10, 3, 81));
    printf("intervened((10, 4, 75)):\t%d\t(should be: 1)\n", intervened(10, 4, 75));
    printf("intervened((3, 3, 3)):\t%d\t(should be: 0)\n", intervened(3, 3, 3));
    printf("intervened((10, 3, 10)):\t%d\t(should be: 0)\n", intervened(10, 3, 10));

    printf("\n");

    printf("nondeterministic_shielded_strategy((3, 3, 37)):\t%d\t(should be: 2)\n", nondeterministic_shielded_strategy(3, 3, 37));
    printf("nondeterministic_shielded_strategy((3, 3, 41)):\t%d\t(should be: 2)\n", nondeterministic_shielded_strategy(3, 3, 41));
    printf("nondeterministic_shielded_strategy((10, 3, 81)):\t%d\t(should be: 1)\n", nondeterministic_shielded_strategy(10, 3, 81));
    printf("nondeterministic_shielded_strategy((10, 4, 75)):\t%d\t(should be: 1)\n", nondeterministic_shielded_strategy(10, 4, 75));
    printf("nondeterministic_shielded_strategy((3, 3, 3)):\t%d\t(should be: 1)\n", nondeterministic_shielded_strategy(3, 3, 3));
    printf("nondeterministic_shielded_strategy((10, 3, 10)):\t%d\t(should be: 1)\n", nondeterministic_shielded_strategy(10, 3, 10));
    printf("nondeterministic_shielded_strategy((4, -9, 82)):\t%d\t(should be: 1)\n", nondeterministic_shielded_strategy(4, -9, 82));
    printf("deterministic_shielded_strategy((4, -9, 82)):\t%d\t(should be: 1)\n", deterministic_shielded_strategy(4, -9, 82));

    //jl_atexit_hook(0); // This is strongly recommended, apparently. Unfortunately, UPPAAL doesn't have an exit hook. Lucky I don't do file writes.
}