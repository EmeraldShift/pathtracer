#pragma warning (disable: 4786)

#ifndef __PARSER_H__

#define __PARSER_H__

#include <string>
#include <map>

#include "ParserException.h"
#include "Tokenizer.h"

#include <utility>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <list>

#include "../scene/scene.h"
#include "../scene/transform.h"
#include "../scene/sphere.h"

/*
  class Parser:
    The Parser is where most of the heavy lifting in parsing
    goes.  This particular parser reads in a stream of tokens
    from the Tokenizer and converts them into a scene in
    memory.

    If you really want to know, this parser is written
    as a top-down parser with one symbol of lookahead.
    See the docs on the website if you're interested in
    modifying this somehow.
*/

class Parser {
public:
    // We need the path for referencing files from the
    // base file.
    Parser(Tokenizer &tokenizer, string basePath)
            : _tokenizer(tokenizer), _basePath(std::move(basePath)) {}

    // Parse the top-level scene
    Scene *parseScene();

private:

    // Highest level parsing routines
    void parseTransformableElement(Scene *scene, TransformNode *transform, const Material &mat);

    void parseGroup(Scene *scene, TransformNode *transform, const Material &mat);

    void parseCamera(Scene *scene);

    void parseGeometry(Scene *scene, TransformNode *transform, const Material &mat);


    // Parse lights
    void parsePointLight(Scene *scene);

    void parseDirectionalLight();

    void parseAmbientLight(Scene *sparseparscene);

    // Parse geometry
    void parseSphere(Scene *scene, TransformNode *transform, const Material &mat);

    void parseBox(Scene *scene, TransformNode *transform, const Material &mat);

    void parseSquare(Scene *scene, TransformNode *transform, const Material &mat);

    void parseCylinder(Scene *scene);

    void parseCone(Scene *scene, const Material &mat);

    void parseTrimesh(Scene *scene, TransformNode *transform, const Material &mat);

    void parseFaces(std::list<f4> &faces);

    // Parse transforms
    void parseTranslate(Scene *scene, TransformNode *transform, const Material &mat);

    void parseRotate(Scene *scene, TransformNode *transform, const Material &mat);

    void parseScale(Scene *scene, TransformNode *transform, const Material &mat);

    void parseTransform(Scene *scene, TransformNode *transform, const Material &mat);

    // Helper functions for parsing expressions of the form:
    //   keyword = value;
    float parseScalarExpression();

    f4 parseVec3dExpression();

    f4 parseVec4dExpression();

    bool parseBooleanExpression();

    Material parseMaterialExpression(Scene *scene, const Material &mat);

    std::string parseIdentExpression();

    f4 parseVec3dMaterialParameter(Scene *scene);

    float parseScalarMaterialParameter(Scene *scene);

    // Helper functions for parsing things like vectors
    // and idents.
    float parseScalar();

    std::list<float> parseScalarList();

    f4 parseVec3d();

    f4 parseVec4d();

    bool parseBoolean();

    Material parseMaterial(Scene *scene, const Material &parent);

    std::string parseIdent();

private:
    Tokenizer &_tokenizer;
    std::map<std::string, Material> materials;
    std::string _basePath;
};

#endif


