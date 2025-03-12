#pragma once

#ifdef callback_api_tracing_client_EXPORTS
#    define CLIENT_API __attribute__((visibility("default")))
#else
#    define CLIENT_API
#endif

namespace rocm_api_client
{
void
setup() CLIENT_API;

void
shutdown() CLIENT_API;

void
start() CLIENT_API;

void
stop() CLIENT_API;
}  // namespace rocm_api_client
