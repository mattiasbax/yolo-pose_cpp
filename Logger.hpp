#pragma once

#include <format>
#include <iostream>
#include <source_location>
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

    virtual void Log(
        Priority prio, const std::string& msg, const std::source_location& sl = std::source_location::current( )
    ) const = 0;

protected:
    Priority mVerbosity;
};

class CoutLogger final : public ILogger {
public:
    CoutLogger( ) : ILogger( Priority::Warning ) { }

    CoutLogger( Priority verbosity ) : ILogger( verbosity ) { }

    void Log( Priority prio, const std::string& msg, const std::source_location& sl = std::source_location::current( ) )
        const override
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

        std::cout << std::format( "{}:{} {} {}\n", sl.file_name( ), sl.line( ), prioString, msg );
    }
};
} // namespace Logger