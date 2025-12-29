#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main() {
    char buffer[256];
    const char* response = "{\"observation\":{\"stage\":1}}\n";
    size_t response_len = strlen(response);

    // Redirect stdout to stderr for debug, use fd 3 for output
    int original_stdout = dup(STDOUT_FILENO);
    dup2(STDERR_FILENO, STDOUT_FILENO);

    while (1) {
        ssize_t bytes_read = read(STDIN_FILENO, buffer, 256);
        if (bytes_read <= 0) break;

        write(original_stdout, response, response_len);
    }

    close(original_stdout);
    return 0;
}
