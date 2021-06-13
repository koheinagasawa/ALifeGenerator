/*
* Node.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/NeuralNetwork/Node.h>

DefaultNode::DefaultNode(Type type)
    : m_type(type)
{
}

float DefaultNode::getValue() const
{
    return m_value;
}

void DefaultNode::setValue(float value)
{
    m_value = m_activation ? m_activation->activate(value) : value;
}
