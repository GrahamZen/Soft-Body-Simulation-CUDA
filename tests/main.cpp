#define CATCH_CONFIG_MAIN  // 让 Catch 提供 main()
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

int main(int argc, char** argv)
{
    Catch::Session session; // There must be exactly one instance

    // Build a new parser on top of Catch's
    using namespace Catch::Clara;
    auto cli = session.cli();

    session.cli(cli);

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    return session.run();
}