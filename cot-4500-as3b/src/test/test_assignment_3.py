{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
from main.assignment_3 import question_1, question_2, question_3, question_4\
\
def test_question_1():\
    x = question_1()\
    expected = np.array([1.2446381, 1.25131659, 2.0])  # Approximate values\
    assert np.allclose(x, expected, rtol=1e-5)\
\
def test_question_2():\
    det, L, U = question_2()\
    assert np.isclose(det, -13.0)\
    assert L.shape == (4, 4)\
    assert U.shape == (4, 4)\
\
def test_question_3():\
    assert question_3() == True\
\
def test_question_4():\
    assert question_4() == True}