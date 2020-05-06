#pragma warning (disable: 4786)

#include <iostream>
#include <sstream>

#include "Parser.h"
#include "../scene/scene.h"
#include "../scene/sphere.h"
#include "../scene/trimesh.h"
#include <glm/gtx/transform.hpp>

// Texture cache
std::map<std::string, TextureMap *> textureCache;

static TextureMap *getTexture(string name) {
    auto itr = textureCache.find(name);
    if (itr == textureCache.end())
        return textureCache[name] = new TextureMap(name);
    return itr->second;
}

Scene *Parser::parseScene() {
    _tokenizer.Read(SBT_RAYTRACER);

    unique_ptr<Token> versionNumber(_tokenizer.Read(SCALAR));

    if (versionNumber->value() > 1.1) {
        std::ostringstream ost;
        ost << "SBT-raytracer version number " << versionNumber->value() <<
            " too high; only able to parse v1.1 and below.";
        throw ParserException(ost.str());
    }

    auto scene = new Scene;
    Material mat;
    TransformRoot transformRoot;

    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case SPHERE:
            case BOX:
            case SQUARE:
            case CYLINDER:
            case CONE:
            case TRIMESH:
            case TRANSLATE:
            case ROTATE:
            case SCALE:
            case TRANSFORM:
            case LBRACE:
                parseTransformableElement(scene, &transformRoot, mat);
                break;
            case POINT_LIGHT:
                parsePointLight(scene);
                break;
            case DIRECTIONAL_LIGHT:
                parseDirectionalLight();
                break;
            case AMBIENT_LIGHT:
                parseAmbientLight(scene);
                break;
            case CAMERA:
                parseCamera(scene);
                break;
            case MATERIAL: {
                mat = parseMaterialExpression(scene, mat);
            }
                break;
            case SEMICOLON:
                _tokenizer.Read(SEMICOLON);
                break;
            case EOFSYM:
                return scene;
            default:
                throw SyntaxErrorException("Expected: geometry, camera, or light information", _tokenizer);
        }
    }
}

void Parser::parseCamera(Scene *scene) {
    bool hasViewDir(false), hasUpDir(false);
    glm::dvec3 viewDir, upDir;

    _tokenizer.Read(CAMERA);
    _tokenizer.Read(LBRACE);

    for (;;) {
        const Token *t = _tokenizer.Peek();

        glm::dvec4 quaternion;
        switch (t->kind()) {
            case POSITION:
                scene->getCamera().setEye(parseVec3dExpression());
                break;
            case FOV:
                scene->getCamera().setFOV(parseScalarExpression());
                break;
            case QUATERNIAN:
                quaternion = parseVec4dExpression();
                scene->getCamera().setLook(
                        quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
                break;
            case ASPECTRATIO:
                scene->getCamera().setAspectRatio(parseScalarExpression());
                break;
            case VIEWDIR:
                viewDir = parseVec3dExpression();
                hasViewDir = true;
                break;
            case UPDIR:
                upDir = parseVec3dExpression();
                hasUpDir = true;
                break;
            case RBRACE:
                // semantic checks
                if (hasViewDir) {
                    if (!hasUpDir)
                        throw SyntaxErrorException("Expected: 'updir'", _tokenizer);
                    scene->getCamera().setLook(viewDir, upDir);
                } else {
                    if (hasUpDir)
                        throw SyntaxErrorException("Expected: 'viewdir'", _tokenizer);
                }
                _tokenizer.Read(RBRACE);
                return;
            default:
                throw SyntaxErrorException("Expected: camera attribute", _tokenizer);
        }
    }
}

void Parser::parseTransformableElement(Scene *scene, TransformNode *transform, const Material &mat) {
    const Token *t = _tokenizer.Peek();
    switch (t->kind()) {
        case SPHERE:
        case BOX:
        case SQUARE:
        case CYLINDER:
        case CONE:
        case TRIMESH:
        case TRANSLATE:
        case ROTATE:
        case SCALE:
        case TRANSFORM:
            parseGeometry(scene, transform, mat);
            break;
        case LBRACE:
            parseGroup(scene, transform, mat);
            break;
        default:
            throw SyntaxErrorException("Expected: transformable element", _tokenizer);
    }
}

// parse a group of geometry, i.e., enclosed in {} blocks.
void Parser::parseGroup(Scene *scene, TransformNode *transform, const Material &mat) {
    Material newMat = mat;
    _tokenizer.Read(LBRACE);
    for (;;) {
        const Token *t = _tokenizer.Peek();
        switch (t->kind()) {
            case SPHERE:
            case BOX:
            case SQUARE:
            case CYLINDER:
            case CONE:
            case TRIMESH:
            case TRANSLATE:
            case ROTATE:
            case SCALE:
            case TRANSFORM:
            case LBRACE:
                parseTransformableElement(scene, transform, newMat);
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                return;
            case MATERIAL: {
                std::cout << "mater" << std::endl;
                newMat = parseMaterialExpression(scene, mat);
            }
            default:
                throw SyntaxErrorException("Expected: '}' or geometry", _tokenizer);
        }
    }
}


void Parser::parseGeometry(Scene *scene, TransformNode *transform, const Material &mat) {
    const Token *t = _tokenizer.Peek();
    switch (t->kind()) {
        case SPHERE:
            parseSphere(scene, transform, mat);
            return;
        case BOX:
            parseBox(scene, transform, mat);
            return;
        case SQUARE:
            parseSquare(scene, transform, mat);
            return;
        case CYLINDER:
            parseCylinder();
            return;
        case CONE:
            parseCone(scene, mat);
            return;
        case TRIMESH:
            parseTrimesh(scene, transform, mat);
            return;
        case TRANSLATE:
            parseTranslate(scene, transform, mat);
            return;
        case ROTATE:
            parseRotate(scene, transform, mat);
            return;
        case SCALE:
            parseScale(scene, transform, mat);
            return;
        case TRANSFORM:
            parseTransform(scene, transform, mat);
            return;
        default:
            throw ParserFatalException("Unrecognized geometry type.");
    }
}


void Parser::parseTranslate(Scene *scene, TransformNode *transform, const Material &mat) {
    _tokenizer.Read(TRANSLATE);
    _tokenizer.Read(LPAREN);
    double x = parseScalar();
    _tokenizer.Read(COMMA);
    double y = parseScalar();
    _tokenizer.Read(COMMA);
    double z = parseScalar();
    _tokenizer.Read(COMMA);

    // Parse child geometry
    parseTransformableElement(scene,
                              transform->createChild(glm::translate(glm::dvec3(x, y, z))), mat);

    _tokenizer.Read(RPAREN);
    _tokenizer.CondRead(SEMICOLON);

    return;
}

void Parser::parseRotate(Scene *scene, TransformNode *transform, const Material &mat) {
    _tokenizer.Read(ROTATE);
    _tokenizer.Read(LPAREN);
    double x = parseScalar();
    _tokenizer.Read(COMMA);
    double y = parseScalar();
    _tokenizer.Read(COMMA);
    double z = parseScalar();
    _tokenizer.Read(COMMA);
    double w = parseScalar();
    _tokenizer.Read(COMMA);

    // Parse child geometry
    parseTransformableElement(scene,
                              transform->createChild(glm::rotate(w, glm::dvec3(x, y, z))), mat);

    _tokenizer.Read(RPAREN);
    _tokenizer.CondRead(SEMICOLON);

    return;
}


void Parser::parseScale(Scene *scene, TransformNode *transform, const Material &mat) {
    _tokenizer.Read(SCALE);
    _tokenizer.Read(LPAREN);
    double x, y, z;

    x = parseScalar();
    _tokenizer.Read(COMMA);

    const Token *next = _tokenizer.Peek();
    if (SCALAR == next->kind()) {
        y = parseScalar();
        _tokenizer.Read(COMMA);
        z = parseScalar();
        _tokenizer.Read(COMMA);
    } else {
        y = x;
        z = x;
    }

    // Parse child geometry
    parseTransformableElement(scene,
                              transform->createChild(glm::scale(glm::dvec3(x, y, z))), mat);

    _tokenizer.Read(RPAREN);
    _tokenizer.CondRead(SEMICOLON);

    return;
}


void Parser::parseTransform(Scene *scene, TransformNode *transform, const Material &mat) {
    _tokenizer.Read(TRANSFORM);
    _tokenizer.Read(LPAREN);

    glm::dvec4 row1 = parseVec4d();
    _tokenizer.Read(COMMA);
    glm::dvec4 row2 = parseVec4d();
    _tokenizer.Read(COMMA);
    glm::dvec4 row3 = parseVec4d();
    _tokenizer.Read(COMMA);
    glm::dvec4 row4 = parseVec4d();
    _tokenizer.Read(COMMA);

    parseTransformableElement(scene,
                              transform->createChild(glm::transpose(glm::dmat4x4(row1, row2, row3, row4))), mat);

    _tokenizer.Read(RPAREN);
    _tokenizer.CondRead(SEMICOLON);

    return;
}

void Parser::parseSphere(Scene *scene, TransformNode *transform, const Material &mat) {
    Sphere *sphere;
    Material newMat = mat;

    _tokenizer.Read(SPHERE);
    _tokenizer.Read(LBRACE);

    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case MATERIAL:
                newMat = parseMaterialExpression(scene, mat);
                break;
            case NAME:
                parseIdentExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                scene->add(new Sphere(newMat,
                                      transform->localToGlobalCoords(glm::dvec4(0, 0, 0, 1)),
                                      transform->localToGlobalCoords(glm::dvec4(1, 0, 0, 0)).x));
                return;
            default:
                throw SyntaxErrorException("Expected: sphere attributes", _tokenizer);
        }
    }
}

void Parser::parseBox(Scene *scene, TransformNode *transform, const Material &mat) {
    Material newMat = mat;

    _tokenizer.Read(BOX);
    _tokenizer.Read(LBRACE);

    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case MATERIAL:
                newMat = parseMaterialExpression(scene, mat);
                break;
            case NAME:
                parseIdentExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                scene->add(Trimesh::fromBox(newMat, transform)->getFaces());
                return;
            default:
                throw SyntaxErrorException("Expected: box attributes", _tokenizer);
        }
    }
}

void Parser::parseSquare(Scene *scene, TransformNode *transform, const Material &mat) {
    Material newMat = mat;

    _tokenizer.Read(SQUARE);
    _tokenizer.Read(LBRACE);

    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case MATERIAL:
                newMat = parseMaterialExpression(scene, mat);
                break;
            case NAME:
                parseIdentExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                scene->add(Trimesh::fromSquare(newMat, transform)->getFaces());
                return;
            default:
                throw SyntaxErrorException("Expected: square attributes", _tokenizer);
        }
    }
}

void Parser::parseCylinder() {
    _tokenizer.Read(CYLINDER);
    _tokenizer.Read(LBRACE);
    for (;;) {
        const Token *t = _tokenizer.Peek();
        switch (t->kind()) {
            case MATERIAL:
                break;
            case NAME:
                parseIdentExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                return;
            default:
                throw SyntaxErrorException("Expected: cylinder attributes", _tokenizer);
        }
    }

}

void Parser::parseCone(Scene *scene, const Material &mat) {
    _tokenizer.Read(CONE);
    _tokenizer.Read(LBRACE);
    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case MATERIAL:
                parseMaterialExpression(scene, mat);
                break;
            case NAME:
                parseIdentExpression();
                break;
            case CAPPED:
                parseBooleanExpression();
                break;
            case BOTTOM_RADIUS:
            case TOP_RADIUS:
            case HEIGHT:
                parseScalarExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                return;
            default:
                throw SyntaxErrorException("Expected: cone attributes", _tokenizer);
        }
    }
}

void Parser::parseTrimesh(Scene *scene, TransformNode *transform, const Material &mat) {
    auto mesh = new Trimesh(mat);

    _tokenizer.Read(TRIMESH);
    _tokenizer.Read(LBRACE);

    bool generateNormals(false);
    std::list<glm::dvec3> faces;

    const char *error;
    for (;;) {
        const Token *t = _tokenizer.Peek();

        switch (t->kind()) {
            case GENNORMALS:
                _tokenizer.Read(GENNORMALS);
                _tokenizer.Read(SEMICOLON);
                generateNormals = true;
                break;

            case MATERIAL:
                mesh->setMaterial(parseMaterialExpression(scene, mat));
                break;

            case NAME:
                parseIdentExpression();
                break;

            case MATERIALS:
                _tokenizer.Read(MATERIALS);
                _tokenizer.Read(EQUALS);
                _tokenizer.Read(LPAREN);
                if (RPAREN != _tokenizer.Peek()->kind()) {
                    mesh->addMaterial(parseMaterial(scene, mesh->getMaterial()));
                    for (;;) {
                        const Token *nextToken = _tokenizer.Peek();
                        if (RPAREN == nextToken->kind())
                            break;
                        _tokenizer.Read(COMMA);
                        mesh->addMaterial(parseMaterial(scene, mesh->getMaterial()));
                    }
                }
                _tokenizer.Read(RPAREN);
                _tokenizer.Read(SEMICOLON);
                break;

            case NORMALS:
                _tokenizer.Read(NORMALS);
                _tokenizer.Read(EQUALS);
                _tokenizer.Read(LPAREN);
                if (RPAREN != _tokenizer.Peek()->kind()) {
                    mesh->addNormal(transform->localToGlobalCoordsNormal(parseVec3d()));
                    for (;;) {
                        const Token *nextToken = _tokenizer.Peek();
                        if (RPAREN == nextToken->kind())
                            break;
                        _tokenizer.Read(COMMA);
                        mesh->addNormal(transform->localToGlobalCoordsNormal(parseVec3d()));
                    }
                }
                _tokenizer.Read(RPAREN);
                _tokenizer.Read(SEMICOLON);
                break;

            case FACES:
                _tokenizer.Read(FACES);
                _tokenizer.Read(EQUALS);
                _tokenizer.Read(LPAREN);
                if (RPAREN != _tokenizer.Peek()->kind()) {
                    parseFaces(faces);
                    for (;;) {
                        const Token *nextToken = _tokenizer.Peek();
                        if (RPAREN == nextToken->kind())
                            break;
                        _tokenizer.Read(COMMA);
                        parseFaces(faces);
                    }
                }
                _tokenizer.Read(RPAREN);
                _tokenizer.Read(SEMICOLON);
                break;

            case POLYPOINTS:
                _tokenizer.Read(POLYPOINTS);
                _tokenizer.Read(EQUALS);
                _tokenizer.Read(LPAREN);
                if (RPAREN != _tokenizer.Peek()->kind()) {
                    mesh->addVertex(transform->localToGlobalCoords(parseVec3d()));
                    for (;;) {
                        const Token *nextToken = _tokenizer.Peek();
                        if (RPAREN == nextToken->kind())
                            break;
                        _tokenizer.Read(COMMA);
                        mesh->addVertex(transform->localToGlobalCoords(parseVec3d()));
                    }
                }
                _tokenizer.Read(RPAREN);
                _tokenizer.Read(SEMICOLON);
                break;


            case RBRACE: {
                _tokenizer.Read(RBRACE);

                // Now add all the faces into the trimesh, since hopefully
                // the vertices have been parsed out
                for (const auto &face : faces) {
                    if (!mesh->addFace(face[0], face[1], face[2])) {
                        std::ostringstream oss;
                        oss << "Bad face in trimesh: (" << face[0] << ", " << face[1] <<
                            ", " << face[2] << ")";
                        throw ParserException(oss.str());
                    }
                }

                if (generateNormals)
                    mesh->generateNormals();

                if ((error = mesh->doubleCheck()))
                    throw ParserException(error);

                scene->add(mesh->getFaces());
                return;
            }

            default:
                throw SyntaxErrorException("Expected: trimesh attributes", _tokenizer);
        }
    }
}

void Parser::parseFaces(std::list<glm::dvec3> &faces) {
    std::list<double> points = parseScalarList();

    // triangulate here and now.  assume the poly is
    // concave (convex?) and we can triangulate using an arbitrary fan
    if (points.size() < 3)
        throw SyntaxErrorException("Faces must have at least 3 vertices.", _tokenizer);

    std::list<double>::const_iterator i = points.begin();
    double a = (*i++);
    double b = (*i++);
    while (i != points.end()) {
        double c = (*i++);
        faces.emplace_back(glm::dvec3(a, b, c));
        b = c;
    }
}

// Ambient lights are a bit special in that we don't actually
// create a separate Light for each ambient light; instead
// we simply sum all the ambient intensities and put them in
// the scene as the I_a coefficient.
void Parser::parseAmbientLight(Scene *scene) {
    _tokenizer.Read(AMBIENT_LIGHT);
    _tokenizer.Read(LBRACE);
    if (_tokenizer.Peek()->kind() != COLOR)
        throw SyntaxErrorException("Expected color attribute", _tokenizer);

    parseVec3dExpression();
    _tokenizer.Read(RBRACE);
}

void Parser::parsePointLight(Scene *scene) {
    glm::dvec3 position;
    glm::dvec3 color;

    bool hasPosition(false), hasColor(false);

    _tokenizer.Read(POINT_LIGHT);
    _tokenizer.Read(LBRACE);

    for (;;) {
        const Token *t = _tokenizer.Peek();
        switch (t->kind()) {
            case POSITION:
                if (hasPosition)
                    throw SyntaxErrorException("Repeated 'position' attribute", _tokenizer);
                position = parseVec3dExpression();
                hasPosition = true;
                break;

            case COLOR:
                if (hasColor)
                    throw SyntaxErrorException("Repeated 'color' attribute", _tokenizer);
                color = parseVec3dExpression();
                hasColor = true;
                break;

            case CONSTANT_ATTENUATION_COEFF:
            case LINEAR_ATTENUATION_COEFF:
            case QUADRATIC_ATTENUATION_COEFF:
                parseScalarExpression();
                break;

            case RBRACE: {
                if (!hasColor)
                    throw SyntaxErrorException("Expected: 'color'", _tokenizer);
                if (!hasPosition)
                    throw SyntaxErrorException("Expected: 'position'", _tokenizer);
                _tokenizer.Read(RBRACE);
                auto m = Material(color, glm::dvec3(), glm::dvec3(), glm::dvec3(1), 1);
                auto s = new Sphere(m, position, 1);
                scene->add(s);
                return;
            }
            default:
                throw SyntaxErrorException(
                        "expecting 'position' or 'color' attribute, or 'constant_attenuation_coeff', 'linear_attenuation_coeff', or 'quadratic_attenuation_coeff'",
                        _tokenizer);
        }
    }
}

void Parser::parseDirectionalLight() {
    _tokenizer.Read(DIRECTIONAL_LIGHT);
    _tokenizer.Read(LBRACE);
    for (;;) {
        const Token *t = _tokenizer.Peek();
        switch (t->kind()) {
            case DIRECTION:
            case COLOR:
                parseVec3dExpression();
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                return;
            default:
                throw SyntaxErrorException("expecting 'position' or 'color' attribute",
                                           _tokenizer);
        }
    }
}

// These ought to be done with template member functions, but compiler support for
// these is rather iffy...
double Parser::parseScalarExpression() {
    // Throw out first token, which precedes the = sign
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    double value(parseScalar());
    _tokenizer.CondRead(SEMICOLON);
    return value;
}

bool Parser::parseBooleanExpression() {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    bool value(parseBoolean());
    _tokenizer.CondRead(SEMICOLON);
    return value;
}

glm::dvec3 Parser::parseVec3dExpression() {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    glm::dvec3 value(parseVec3d());
    _tokenizer.CondRead(SEMICOLON);
    return value;
}

glm::dvec4 Parser::parseVec4dExpression() {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    glm::dvec4 value(parseVec4d());
    _tokenizer.CondRead(SEMICOLON);
    return value;
}

Material Parser::parseMaterialExpression(Scene *scene, const Material &parent) {
    _tokenizer.Read(MATERIAL);
    _tokenizer.Read(EQUALS);
    Material mat = parseMaterial(scene, parent);
    _tokenizer.CondRead(SEMICOLON);
    return mat;
}

string Parser::parseIdentExpression() {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    string value(parseIdent());
    _tokenizer.CondRead(SEMICOLON);
    return value;
}

double Parser::parseScalar() {
    unique_ptr<Token> scalar(_tokenizer.Read(SCALAR));

    return scalar->value();
}

string Parser::parseIdent() {
    unique_ptr<Token> scalar(_tokenizer.Read(IDENT));

    return scalar->ident();
}


std::list<double> Parser::parseScalarList() {
    std::list<double> ret;

    _tokenizer.Read(LPAREN);
    if (RPAREN != _tokenizer.Peek()->kind()) {
        ret.push_back(parseScalar());
        for (;;) {
            const Token *nextToken = _tokenizer.Peek();
            if (RPAREN == nextToken->kind())
                break;
            _tokenizer.Read(COMMA);
            ret.push_back(parseScalar());
        }
    }
    _tokenizer.Read(RPAREN);

    return ret;

}

bool Parser::parseBoolean() {
    const Token *next = _tokenizer.Peek();
    if (SYMTRUE == next->kind()) {
        _tokenizer.Read(SYMTRUE);
        return true;
    }
    if (SYMFALSE == next->kind()) {
        _tokenizer.Read(SYMFALSE);
        return false;
    }
    throw SyntaxErrorException("Expected boolean", _tokenizer);
}

glm::dvec3 Parser::parseVec3d() {
    _tokenizer.Read(LPAREN);
    unique_ptr<Token> value1(_tokenizer.Read(SCALAR));
    _tokenizer.Read(COMMA);
    unique_ptr<Token> value2(_tokenizer.Read(SCALAR));
    _tokenizer.Read(COMMA);
    unique_ptr<Token> value3(_tokenizer.Read(SCALAR));
    _tokenizer.Read(RPAREN);

    return glm::dvec3(value1->value(),
                      value2->value(),
                      value3->value());
}

glm::dvec4 Parser::parseVec4d() {
    _tokenizer.Read(LPAREN);
    unique_ptr<Token> value1(_tokenizer.Read(SCALAR));
    _tokenizer.Read(COMMA);
    unique_ptr<Token> value2(_tokenizer.Read(SCALAR));
    _tokenizer.Read(COMMA);
    unique_ptr<Token> value3(_tokenizer.Read(SCALAR));
    _tokenizer.Read(COMMA);
    unique_ptr<Token> value4(_tokenizer.Read(SCALAR));
    _tokenizer.Read(RPAREN);

    return glm::dvec4(value1->value(),
                      value2->value(),
                      value3->value(),
                      value4->value());
}

Material Parser::parseMaterial(Scene *scene, const Material &parent) {
    const Token *tok = _tokenizer.Peek();
    if (IDENT == tok->kind()) {
        return materials[tok->ident()];
    }

    _tokenizer.Read(LBRACE);

    string name;

    auto mat = Material(parent);
    for (;;) {
        const Token *token = _tokenizer.Peek();
        switch (token->kind()) {
            case EMISSIVE:
                mat.setEmissive(parseVec3dMaterialParameter(scene));
                break;
            case AMBIENT:
                parseVec3dMaterialParameter(scene);
                break;
            case SPECULAR: {
                parseVec3dMaterialParameter(scene);
                break;
            }
            case DIFFUSE:
                mat.setDiffuse(parseVec3dMaterialParameter(scene));
                break;

            case REFLECTIVE:
                mat.setReflective(parseVec3dMaterialParameter(scene));
                break;
            case TRANSMISSIVE:
                mat.setTransmissive(parseVec3dMaterialParameter(scene));
                break;
            case INDEX:
                mat.setIndex(parseScalarMaterialParameter(scene));
                break;
            case SHININESS:
                parseScalarMaterialParameter(scene);
                break;
            case NAME:
                _tokenizer.Read(NAME);
                name = (_tokenizer.Read(IDENT))->ident();
                _tokenizer.Read(SEMICOLON);
                break;
            case RBRACE:
                _tokenizer.Read(RBRACE);
                if (!name.empty()) {
                    if (materials.find(name) == materials.end())
                        materials[name] = mat;
                    else {
                        std::ostringstream oss;
                        oss << "Redefinition of material '" << name << "'.";
                        throw SyntaxErrorException(oss.str(), _tokenizer);
                    }
                }
                return mat;
            default:
                throw SyntaxErrorException("Expected: material attribute", _tokenizer);
        }
    }
}

MaterialParameter Parser::parseVec3dMaterialParameter(Scene *scene) {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    if (_tokenizer.CondRead(MAP)) {
        _tokenizer.Read(LPAREN);
        string filename = _basePath;
        filename.append("/");
        filename.append(parseIdent());
        _tokenizer.Read(RPAREN);
        _tokenizer.CondRead(SEMICOLON);
        return MaterialParameter(getTexture(filename));
    } else {
        glm::dvec3 value(parseVec3d());
        _tokenizer.CondRead(SEMICOLON);
        return MaterialParameter(value);
    }
}

MaterialParameter Parser::parseScalarMaterialParameter(Scene *scene) {
    _tokenizer.Get();
    _tokenizer.Read(EQUALS);
    if (_tokenizer.CondRead(MAP)) {
        _tokenizer.Read(LPAREN);
        string filename = parseIdent();
        _tokenizer.Read(RPAREN);
        _tokenizer.CondRead(SEMICOLON);
        return MaterialParameter(getTexture(filename));
    } else {
        double value = parseScalar();
        _tokenizer.CondRead(SEMICOLON);
        return MaterialParameter(value);
    }
}
