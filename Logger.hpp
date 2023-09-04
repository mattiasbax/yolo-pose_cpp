#pragma once

#include <format>
#include <iostream>
#include <string>

namespace Logger {

enum class Priority {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Fatal
};

class ILogger {
public:
    ILogger( ) : mVerbosity( Priority::Warning ) { }

    ILogger( Priority verbosity ) : mVerbosity( verbosity ) { }

    virtual ~ILogger( ) = default;

    void SetVerbosity( Priority verbosity ) { mVerbosity = verbosity; }

    virtual void Log( Priority prio, const std::string& msg ) const = 0;

protected:
    Priority mVerbosity;
};

class CoutLogger final : public ILogger {
public:
    CoutLogger( ) : ILogger( Priority::Warning ) { }

    CoutLogger( Priority verbosity ) : ILogger( verbosity ) { }

    void Log( Priority prio, const std::string& msg ) const override
    {
        if ( prio < mVerbosity )
            return;

        std::string prioString;
        switch ( prio ) {
        case Priority::Debug:
            prioString = "[DEBUG]";
            break;
        case Priority::Info:
            prioString = "[INFO]";
            break;
        case Priority::Warning:
            prioString = "[WARNING]";
            break;
        case Priority::Error:
            prioString = "[ERROR]";
            break;
        case Priority::Critical:
            prioString = "[CRITICAL]";
            break;
        case Priority::Fatal:
            prioString = "[FATAL]";
            break;
        default:
            break;
        }

        std::cout << std::format( "{}:{} {} {}\n", __FILE__, __LINE__, prioString, msg );
    }
};
} // namespace Logger