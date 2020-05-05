#ifndef __PARSEREXCEPTION_H__

#define __PARSEREXCEPTION_H__

#include <string>
#include <utility>

/* These are some exceptions used by the parser class. 
   You shouldn't need to mess with them. 
*/

using std::string;
using std::ostream;


class Tokenizer;

class ParserException {
public:
    explicit ParserException(std::string msg)
            : _msg(std::move(msg)) {}

    const std::string &message() const { return _msg; }

private:
    const std::string _msg;

};

class ParserFatalException
        : public ParserException {
public:
    explicit ParserFatalException(const std::string &msg)
            : ParserException(msg) {}

private:

};

class SyntaxErrorException
        : public ParserException {
public:
    SyntaxErrorException(const std::string &msg, const Tokenizer &tok);

    string formattedMessage() const { return _formattedMsg; }

private:
    string _formattedMsg;
};

#endif


