#include <pipelink/stage.hpp>
#include <iostream>

int main() {
  try {
    pl::detail::emit(std::cout);
    return std::cout.good() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
