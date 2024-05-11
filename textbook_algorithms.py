{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "cellView": "form",
    "execution": {
     "iopub.execute_input": "2023-09-19T09:07:56.497459Z",
     "iopub.status.busy": "2023-09-19T09:07:56.496916Z",
     "iopub.status.idle": "2023-09-19T09:07:56.501382Z",
     "shell.execute_reply": "2023-09-19T09:07:56.500692Z"
    },
    "id": "906e07f6e562"
   },
   "outputs": [],
   "source": [
    "#@title Copyright 2020 The Cirq Developers\n",
    "# Licensed under the Apache License, Version 2.0 (the \"License\");\n",
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at\n",
    "#\n",
    "# https://www.apache.org/licenses/LICENSE-2.0\n",
    "#\n",
    "# Unless required by applicable law or agreed to in writing, software\n",
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
    "# See the License for the specific language governing permissions and\n",
    "# limitations under the License."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7c5ad5c66a5b"
   },
   "source": [
    "# Textbook algorithms in Cirq"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "15bb25b1082e"
   },
   "source": [
    "<table class=\"tfo-notebook-buttons\" align=\"left\">\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://quantumai.google/cirq/experiments/textbook_algorithms\"><img src=\"https://quantumai.google/site-assets/images/buttons/quantumai_logo_1x.png\" />View on QuantumAI</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/quantumlib/Cirq/blob/master/docs/experiments/textbook_algorithms.ipynb\"><img src=\"https://quantumai.google/site-assets/images/buttons/colab_logo_1x.png\" />Run in Google Colab</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://github.com/quantumlib/Cirq/blob/master/docs/experiments/textbook_algorithms.ipynb\"><img src=\"https://quantumai.google/site-assets/images/buttons/github_logo_1x.png\" />View source on GitHub</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a href=\"https://storage.googleapis.com/tensorflow_docs/Cirq/docs/experiments/textbook_algorithms.ipynb\"><img src=\"https://quantumai.google/site-assets/images/buttons/download_icon_1x.png\" />Download notebook</a>\n",
    "  </td>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lORoela1QICx"
   },
   "source": [
    "In this notebook we'll run through some Cirq implementations of some of the standard algorithms that one encounters in an introductory quantum computing course. In particular, we will discuss the quantum teleportation algorithm, quantum Fourier transform, phase estimation algorithm, and Grover's algorithm. The discussion here is expanded from examples found in the [Cirq examples](https://github.com/quantumlib/Cirq/tree/master/examples) directory."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:07:56.505457Z",
     "iopub.status.busy": "2023-09-19T09:07:56.504865Z",
     "iopub.status.idle": "2023-09-19T09:08:17.279061Z",
     "shell.execute_reply": "2023-09-19T09:08:17.278292Z"
    },
    "id": "pPMSHs4HQfSR"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "installing cirq...\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\r\n",
      "jupyter-events 0.7.0 requires jsonschema[format-nongpl]>=4.18.0, but you have jsonschema 4.17.3 which is incompatible.\r\n",
      "jupyterlab-server 2.25.0 requires jsonschema>=4.18.0, but you have jsonschema 4.17.3 which is incompatible.\r\n",
      "referencing 0.30.2 requires attrs>=22.2.0, but you have attrs 21.4.0 which is incompatible.\r\n",
      "tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 4.24.3 which is incompatible.\u001b[0m\u001b[31m\r\n",
      "\u001b[0m"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "installed cirq.\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "    import cirq\n",
    "except ImportError:\n",
    "    print(\"installing cirq...\")\n",
    "    !pip install cirq --quiet\n",
    "    import cirq\n",
    "    print(\"installed cirq.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.282836Z",
     "iopub.status.busy": "2023-09-19T09:08:17.282261Z",
     "iopub.status.idle": "2023-09-19T09:08:17.286256Z",
     "shell.execute_reply": "2023-09-19T09:08:17.285640Z"
    },
    "id": "57aaba33f657"
   },
   "outputs": [],
   "source": [
    "import random\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "laCvAwThaADq"
   },
   "source": [
    "## Quantum teleportation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xytEjvt9cD5L"
   },
   "source": [
    "Quantum teleportation is a process by which a quantum state can be transmitted by sending only two classical bits of information. This is accomplished by pre-sharing an entangled state between the sender (Alice) and the receiver (Bob). This entangled state allows the receiver (Bob) of the two classical bits of information to possess a qubit with the same state as the one held by the sender (Alice).\n",
    "\n",
    "In the cell below, we define a function which implements the circuit for quantum teleportation. This function inputs a gate which prepares the *message qubit* in some state to transmit from Alice to Bob.\n",
    "\n",
    "> For more background on quantum teleportation or to see the mathematics of why it works, check out [the original paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895) or the [Wikipedia page](https://en.wikipedia.org/wiki/Quantum_teleportation). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.289708Z",
     "iopub.status.busy": "2023-09-19T09:08:17.289067Z",
     "iopub.status.idle": "2023-09-19T09:08:17.295026Z",
     "shell.execute_reply": "2023-09-19T09:08:17.294382Z"
    },
    "id": "Ex8ka640a5xN"
   },
   "outputs": [],
   "source": [
    "def make_quantum_teleportation_circuit(gate):\n",
    "    \"\"\"Returns a circuit for quantum teleportation.\n",
    "    \n",
    "    This circuit 'teleports' a random qubit state prepared by\n",
    "    the input gate from Alice to Bob.\n",
    "    \"\"\"\n",
    "    circuit = cirq.Circuit()\n",
    "    \n",
    "    # Get the three qubits involved in the teleportation protocol.\n",
    "    msg = cirq.NamedQubit(\"Message\")\n",
    "    alice = cirq.NamedQubit(\"Alice\")\n",
    "    bob = cirq.NamedQubit(\"Bob\")\n",
    "    \n",
    "    # The input gate prepares the message to send.\n",
    "    circuit.append(gate(msg))\n",
    "\n",
    "    # Create a Bell state shared between Alice and Bob.\n",
    "    circuit.append([cirq.H(alice), cirq.CNOT(alice, bob)])\n",
    "    \n",
    "    # Bell measurement of the Message and Alice's entangled qubit.\n",
    "    circuit.append([cirq.CNOT(msg, alice), cirq.H(msg), cirq.measure(msg, alice)])\n",
    "\n",
    "    # Uses the two classical bits from the Bell measurement to recover the\n",
    "    # original quantum message on Bob's entangled qubit.\n",
    "    circuit.append([cirq.CNOT(alice, bob), cirq.CZ(msg, bob)])\n",
    "\n",
    "    return circuit"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "715674658f8c"
   },
   "source": [
    "Now, we define a gate to prepare the message qubit in some state, then visualize the teleportation circuit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.298210Z",
     "iopub.status.busy": "2023-09-19T09:08:17.297831Z",
     "iopub.status.idle": "2023-09-19T09:08:17.305562Z",
     "shell.execute_reply": "2023-09-19T09:08:17.304973Z"
    },
    "id": "023602d016d8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Teleportation circuit:\n",
      "\n",
      "Alice: ─────H────────@───X───────M───@───────\n",
      "                     │   │       │   │\n",
      "Bob: ────────────────X───┼───────┼───X───@───\n",
      "                         │       │       │\n",
      "Message: ───X^0.25───────@───H───M───────@───\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Visualize the teleportation circuit.\"\"\"\n",
    "# Gate to put the message qubit in some state to send.\n",
    "gate = cirq.X ** 0.25\n",
    "\n",
    "# Create the teleportation circuit.\n",
    "circuit = make_quantum_teleportation_circuit(gate)\n",
    "print(\"Teleportation circuit:\\n\")\n",
    "print(circuit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "18c3ed975229"
   },
   "source": [
    "As discussed, at the end of the circuit, Bob's qubit will be the state of the message qubit. We can verify this by simulating the circuit. First, we check what the state of the message qubit is given the above `gate`. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.308847Z",
     "iopub.status.busy": "2023-09-19T09:08:17.308398Z",
     "iopub.status.idle": "2023-09-19T09:08:17.315361Z",
     "shell.execute_reply": "2023-09-19T09:08:17.314708Z"
    },
    "id": "d18db1bc5fb2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bloch vector of message qubit:\n",
      "[ 0.    -0.707  0.707]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Display the Bloch vector of the message qubit.\"\"\"\n",
    "message = cirq.Circuit(gate.on(cirq.NamedQubit(\"Message\"))).final_state_vector()\n",
    "message_bloch_vector = cirq.bloch_vector_from_state_vector(message, index=0)\n",
    "print(\"Bloch vector of message qubit:\")\n",
    "print(np.round(message_bloch_vector, 3))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7b59d87ff1ae"
   },
   "source": [
    "Now we simulate the entire teleportation circuit and check what the final state of Bob's qubit is."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.318573Z",
     "iopub.status.busy": "2023-09-19T09:08:17.318021Z",
     "iopub.status.idle": "2023-09-19T09:08:17.326354Z",
     "shell.execute_reply": "2023-09-19T09:08:17.325661Z"
    },
    "id": "4303441fdb1f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bloch vector of Bob's qubit:\n",
      "[ 0.    -0.707  0.707]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Simulate the teleportation circuit and get the final state of Bob's qubit.\"\"\"\n",
    "# Get a simulator.\n",
    "sim = cirq.Simulator()\n",
    "\n",
    "# Simulate the teleportation circuit.\n",
    "result = sim.simulate(circuit)\n",
    "\n",
    "# Get the Bloch vector of Bob's qubit.\n",
    "bobs_bloch_vector = cirq.bloch_vector_from_state_vector(result.final_state_vector, index=1)\n",
    "print(\"Bloch vector of Bob's qubit:\")\n",
    "print(np.round(bobs_bloch_vector, 3))\n",
    "\n",
    "# Verify they are the same state!\n",
    "np.testing.assert_allclose(bobs_bloch_vector, message_bloch_vector, atol=1e-6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b906e52e2080"
   },
   "source": [
    "As we can see, the final state of Bob's qubit is the same as the initial state of the message qubit! One can change the `gate` above and re-run the protocol. The final state of Bob's qubit will always be the initial state of the message qubit."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5c6236bee54c"
   },
   "source": [
    "## Quantum Fourier transform"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LHcyAAtbQBdM"
   },
   "source": [
    "This section provides an overview of the quantum Fourier transform which we use in the next section for the phase estimation algorithm."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7bf118a787f0"
   },
   "source": [
    "### Overview of the quantum Fourier transform"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EKi8ZKeOI8MX"
   },
   "source": [
    "We'll start out by reminding ourselves what the [quantum Fourier transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) does, and how it should be constructed.\n",
    "\n",
    "Suppose we have an $n$-qubit state $|x\\rangle$ where $x$ is an integer in the range $0$ to $2^{n}-1$. (That is, $|x\\rangle$ is a computational basis state.) The quantum Fourier transform (QFT) performs the following operation:\n",
    "\n",
    "$$\n",
    "\\text{QFT}|x\\rangle = \\frac{1}{2^{n/2}} \\sum_{y=0}^{2^n-1} e^{2\\pi i y x/2^n} |y\\rangle.\n",
    "$$\n",
    "\n",
    "> *Note*: The QFT maps from the computational basis to the frequency basis.\n",
    "\n",
    "Though it may not be obvious at first glance, the QFT is actually a unitary transformation. As a matrix, the QFT is given by\n",
    "\n",
    "$$\n",
    "\\text{QFT} = \\begin{bmatrix}\n",
    "1 & 1 & 1& \\cdots &1 \\\\\n",
    "1 & \\omega & \\omega^2& \\cdots &\\omega^{2^n-1} \\\\\n",
    "1 & \\omega^2 & \\omega^4& \\cdots &\\omega^{2(2^n-1)}\\\\\n",
    "\\vdots &\\vdots &\\vdots &\\ddots &\\vdots \\\\\n",
    "1 &\\omega^{2^n-1} &\\omega^{2(2^n-1)} &\\cdots &\\omega^{(2^n-1)(2^n-1)},\n",
    "\\end{bmatrix}\n",
    "$$\n",
    "\n",
    "where $\\omega = e^{2\\pi i /2^n}$. \n",
    "\n",
    "\n",
    "If you believe that the QFT is unitary, then you'll also notice from the matrix form that its inverse is given by a similar expression but with complex-conjugated coefficients:\n",
    "\n",
    "$$\n",
    "\\text{QFT}^{-1}|x\\rangle = \\frac{1}{2^{n/2}} \\sum_{y=0}^{2^n-1} e^{-2\\pi i y x/2^n} |y\\rangle.\n",
    "$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xyhgCosQK3j4"
   },
   "source": [
    "The construction of the QFT as a circuit follows a simple recursive form, though fully justifying it will take us too far from the main goal of this notebook. We really only need to know what the circuit looks like, and for that we can look at the following diagram:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "abfae01ae754"
   },
   "source": [
    "![QFT Circuit](https://upload.wikimedia.org/wikipedia/commons/6/61/Q_fourier_nqubits.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q3fIRQhv6LVp"
   },
   "source": [
    "Here, $x_j$ represents the $j$th bit of the input $x$. The most significant bit is $x_1$ and the least significant bit is $x_n$ so that\n",
    "\n",
    "$$\n",
    "x = \\sum_{j=0}^{n-1} x_{j+1}2^j.\n",
    "$$\n",
    "\n",
    "As usual, $H$ is the Hadamard gate. The Controlled-$R_j$ gates are phase gates similar to the Controlled-$Z$ gate. In fact, for us it will be useful to just think of them as fractional powers of Controlled-$Z$ gates:\n",
    "\n",
    "$$\n",
    "CR_j = CZ^{\\large 1/2^{j-1}}\n",
    "$$\n",
    "\n",
    "Finally, on the far right of the above diagram we have the output representing the bits of $y$. The only difference between the left and right side is that the output bits are in *reversed order*: the most significant bit of $y$ is on the bottom and the least significant bit of $y$ is on the top. One can reverse this by including Swap gates at the end of the circuit."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2dyP_y5AGcWP"
   },
   "source": [
    "### Quantum Fourier transform as a circuit"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RJhjgemX8QXe"
   },
   "source": [
    "Let's define a generator which produces the QFT circuit. It should accept a list of qubits as input and `yield`s the gates to construct the QFT in the right order. A useful observation is that the QFT circuit \"repeats\" smaller versions of itself as you move from left to right across the diagram."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.330421Z",
     "iopub.status.busy": "2023-09-19T09:08:17.329862Z",
     "iopub.status.idle": "2023-09-19T09:08:17.333881Z",
     "shell.execute_reply": "2023-09-19T09:08:17.333244Z"
    },
    "id": "Fu0wP9sLG94Z"
   },
   "outputs": [],
   "source": [
    "def make_qft(qubits):\n",
    "    \"\"\"Generator for the QFT on a list of qubits.\n",
    "    \n",
    "    For four qubits, the answer is:\n",
    "    \n",
    "                          ┌───────┐   ┌────────────┐   ┌───────┐\n",
    "    0: ───H───@────────@───────────@───────────────────────────────────────\n",
    "              │        │           │\n",
    "    1: ───────@^0.5────┼─────H─────┼──────@─────────@──────────────────────\n",
    "                       │           │      │         │\n",
    "    2: ────────────────@^0.25──────┼──────@^0.5─────┼─────H────@───────────\n",
    "                                   │                │          │\n",
    "    3: ────────────────────────────@^(1/8)──────────@^0.25─────@^0.5───H───\n",
    "                      └───────┘   └────────────┘   └───────┘\n",
    "    \"\"\"\n",
    "    # Your code here!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GbFgwEIW83qL"
   },
   "source": [
    "#### Solution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.337111Z",
     "iopub.status.busy": "2023-09-19T09:08:17.336523Z",
     "iopub.status.idle": "2023-09-19T09:08:17.340698Z",
     "shell.execute_reply": "2023-09-19T09:08:17.340083Z"
    },
    "id": "CtDX3krz87eC"
   },
   "outputs": [],
   "source": [
    "def make_qft(qubits):\n",
    "    \"\"\"Generator for the QFT on a list of qubits.\"\"\"\n",
    "    qreg = list(qubits)\n",
    "    while len(qreg) > 0:\n",
    "        q_head = qreg.pop(0)\n",
    "        yield cirq.H(q_head)\n",
    "        for i, qubit in enumerate(qreg):\n",
    "            yield (cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2f0e396fd3f1"
   },
   "source": [
    "We can check the solution agrees with the above diagram by printing it out for a small number of qubits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.343999Z",
     "iopub.status.busy": "2023-09-19T09:08:17.343622Z",
     "iopub.status.idle": "2023-09-19T09:08:17.351405Z",
     "shell.execute_reply": "2023-09-19T09:08:17.350763Z"
    },
    "id": "nhbBPpf9GiHO"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                  ┌───────┐   ┌────────────┐   ┌───────┐\n",
      "0: ───H───@────────@───────────@───────────────────────────────────────\n",
      "          │        │           │\n",
      "1: ───────@^0.5────┼─────H─────┼──────@─────────@──────────────────────\n",
      "                   │           │      │         │\n",
      "2: ────────────────@^0.25──────┼──────@^0.5─────┼─────H────@───────────\n",
      "                               │                │          │\n",
      "3: ────────────────────────────@^(1/8)──────────@^0.25─────@^0.5───H───\n",
      "                  └───────┘   └────────────┘   └───────┘\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Visually check the QFT circuit.\"\"\"\n",
    "qubits = cirq.LineQubit.range(4)\n",
    "qft = cirq.Circuit(make_qft(qubits))\n",
    "print(qft)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "f2dd439b93de"
   },
   "source": [
    "### Quantum Fourier transform as an operation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "316750b4674e"
   },
   "source": [
    "The QFT is such a common subroutine that it is actually a pre-defined gate/operation in Cirq. One can use `cirq.QuantumFourierTransformGate` to get the gate or the helper function `cirq.qft` with a sequence of qubits to get the operation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.354632Z",
     "iopub.status.busy": "2023-09-19T09:08:17.354201Z",
     "iopub.status.idle": "2023-09-19T09:08:17.358501Z",
     "shell.execute_reply": "2023-09-19T09:08:17.357896Z"
    },
    "id": "725d3830c29c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0: ───qft[norev]───\n",
      "      │\n",
      "1: ───#2───────────\n",
      "      │\n",
      "2: ───#3───────────\n",
      "      │\n",
      "3: ───#4───────────\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Use the built-in QFT in Cirq.\"\"\"\n",
    "qft_operation = cirq.qft(*qubits, without_reverse=True)\n",
    "qft_cirq = cirq.Circuit(qft_operation)\n",
    "print(qft_cirq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HgYjV70FfnDr"
   },
   "source": [
    "The function `cirq.qft` has the optional argument `without_reverse` which specifies whether or not to reverse the order of the bits at the end of the circuit. We can confirm the `make_qft` function we defined performs the same transformation as the built-in `cirq.qft` with the following test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.361840Z",
     "iopub.status.busy": "2023-09-19T09:08:17.361408Z",
     "iopub.status.idle": "2023-09-19T09:08:17.367754Z",
     "shell.execute_reply": "2023-09-19T09:08:17.367198Z"
    },
    "id": "i3Ir6kjmDqtt"
   },
   "outputs": [],
   "source": [
    "\"\"\"Check equality of the 'manual' and 'built-in' QFTs.\"\"\"\n",
    "np.testing.assert_allclose(cirq.unitary(qft), cirq.unitary(qft_cirq))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3f318cef3b36"
   },
   "source": [
    "### Inverse quantum Fourier transform as a circuit"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "grSAE3QKf6JB"
   },
   "source": [
    "As mentioned, the only difference between the QFT and inverse QFT is the sign of the exponent of the controlled rotations. Using the `make_qft` function as a guide, complete the `make_qft_inverse` function below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.371144Z",
     "iopub.status.busy": "2023-09-19T09:08:17.370592Z",
     "iopub.status.idle": "2023-09-19T09:08:17.374443Z",
     "shell.execute_reply": "2023-09-19T09:08:17.373898Z"
    },
    "id": "5bcdd1a76fa2"
   },
   "outputs": [],
   "source": [
    "def make_qft_inverse(qubits):\n",
    "    \"\"\"Generator for the inverse QFT on a list of qubits.\n",
    "    \n",
    "    For four qubits, the answer is:\n",
    "    \n",
    "                       ┌────────┐   ┌──────────────┐   ┌────────┐\n",
    "    0: ───H───@─────────@────────────@───────────────────────────────────────────×───\n",
    "              │         │            │                                           │\n",
    "    1: ───────@^-0.5────┼──────H─────┼───────@──────────@────────────────────×───┼───\n",
    "                        │            │       │          │                    │   │\n",
    "    2: ─────────────────@^-0.25──────┼───────@^-0.5─────┼──────H────@────────×───┼───\n",
    "                                     │                  │           │            │\n",
    "    3: ──────────────────────────────@^(-1/8)───────────@^-0.25─────@^-0.5───H───×───\n",
    "                       └────────┘   └──────────────┘   └────────┘\n",
    "    \"\"\"\n",
    "    # Your code here!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8b1c36ce323f"
   },
   "source": [
    "#### Solution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.377231Z",
     "iopub.status.busy": "2023-09-19T09:08:17.376998Z",
     "iopub.status.idle": "2023-09-19T09:08:17.381188Z",
     "shell.execute_reply": "2023-09-19T09:08:17.380579Z"
    },
    "id": "d1c6a6a99bf7"
   },
   "outputs": [],
   "source": [
    "def make_qft_inverse(qubits):\n",
    "    \"\"\"Generator for the inverse QFT on a list of qubits.\"\"\"\n",
    "    qreg = list(qubits)[::-1]\n",
    "    while len(qreg) > 0:\n",
    "        q_head = qreg.pop(0)\n",
    "        yield cirq.H(q_head)\n",
    "        for i, qubit in enumerate(qreg):\n",
    "            yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cf940ffa89a0"
   },
   "source": [
    "We can check the solution agrees with the above diagram by printing it out for a small number of qubits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.384406Z",
     "iopub.status.busy": "2023-09-19T09:08:17.383960Z",
     "iopub.status.idle": "2023-09-19T09:08:17.391612Z",
     "shell.execute_reply": "2023-09-19T09:08:17.390959Z"
    },
    "id": "778b0a8dc5ad"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                   ┌────────┐   ┌──────────────┐   ┌────────┐\n",
      "0: ──────────────────────────────@──────────────────@───────────@────────H───\n",
      "                                 │                  │           │\n",
      "1: ─────────────────@────────────┼───────@──────────┼──────H────@^-0.5───────\n",
      "                    │            │       │          │\n",
      "2: ───────@─────────┼──────H─────┼───────@^-0.5─────@^-0.25──────────────────\n",
      "          │         │            │\n",
      "3: ───H───@^-0.5────@^-0.25──────@^(-1/8)────────────────────────────────────\n",
      "                   └────────┘   └──────────────┘   └────────┘\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Visually check the inverse QFT circuit.\"\"\"\n",
    "qubits = cirq.LineQubit.range(4)\n",
    "iqft = cirq.Circuit(make_qft_inverse(qubits))\n",
    "print(iqft)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "03c3c43b30bc"
   },
   "source": [
    "### Inverse quantum Fourier transform as an operation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "80dddaca5284"
   },
   "source": [
    "The function `cirq.qft` can be used with the optional argument `inverse=True` to return an inverse QFT operation as shown below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.394952Z",
     "iopub.status.busy": "2023-09-19T09:08:17.394433Z",
     "iopub.status.idle": "2023-09-19T09:08:17.399083Z",
     "shell.execute_reply": "2023-09-19T09:08:17.398527Z"
    },
    "id": "c26fb1937ea5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0: ───qft[norev]^-1───\n",
      "      │\n",
      "1: ───#2──────────────\n",
      "      │\n",
      "2: ───#3──────────────\n",
      "      │\n",
      "3: ───#4──────────────\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Use the built-in inverse QFT in Cirq.\"\"\"\n",
    "iqft_operation = cirq.qft(*qubits, inverse=True, without_reverse=True)\n",
    "iqft_cirq = cirq.Circuit(iqft_operation)\n",
    "print(iqft_cirq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b1bd9edb50dd"
   },
   "source": [
    "As above, we can check the `make_qft_inverse` function we defined performs the same transformation as the built-in Cirq function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.402324Z",
     "iopub.status.busy": "2023-09-19T09:08:17.401707Z",
     "iopub.status.idle": "2023-09-19T09:08:17.408883Z",
     "shell.execute_reply": "2023-09-19T09:08:17.408326Z"
    },
    "id": "7ad3bf5b7d38"
   },
   "outputs": [],
   "source": [
    "\"\"\"Check equality of the 'manual' and 'built-in' inverse QFTs.\"\"\"\n",
    "np.testing.assert_allclose(cirq.unitary(iqft), cirq.unitary(iqft_cirq))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3457bc296ce8"
   },
   "source": [
    "## Phase estimation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "r-CjbPwkRI_I"
   },
   "source": [
    "As an application of our quantum Fourier transform circuit, we'll implement the [phase estimation](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) algorithm. The phase estimation algorithm estimates the eigenvalues of a unitary operator and uses the inverse QFT as a subroutine. The total circuit that we are going to implement is shown below."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gJ01TOhr4CQN"
   },
   "source": [
    ">![Phase Estimation](https://upload.wikimedia.org/wikipedia/commons/a/a5/PhaseCircuit-crop.svg)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_15iEUy5Rk1o"
   },
   "source": [
    "Suppose we have a unitary operator $U$ with eigenvector $|\\psi\\rangle$ and eigenvalue $\\exp(2\\pi i \\theta)$. (Every eigenvalue of a unitary can be written this way.) Our objective is to get an $n$-bit approximation to $\\theta$. The first step is to construct the state\n",
    "\n",
    "$$\n",
    "|\\Phi\\rangle = \\frac{1}{2^{n/2}}\\sum_{y=0}^{2^{n-1}} e^{2\\pi i y \\theta}|y\\rangle.\n",
    "$$\n",
    "\n",
    "This looks very similar to the output of the QFT applied to the state $|2^n\\theta\\rangle$, except for the fact that $2^n\\theta$ may not be an integer. If $2^n\\theta$ *were* an integer, then we would apply the inverse QFT and measure the qubits to read off the binary representation of $2^n\\theta$. Even if $2^n\\theta$ is not an integer, we can still perform the same procedure and the result will be a sequence of bits that, with high probability, gives an $n$-bit approximation to $\\theta$. We just have to repeat the procedure a few times to be sure of the answer."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sypcpUzLTxRK"
   },
   "source": [
    "Since we've already constructed the inverse QFT, all we really have to do is figure out how to construct the state $|\\Phi\\rangle$. This is accomplished by the first part of the circuit pictured above. We begin by applying $H^{\\otimes n}$ to the state $|0\\rangle$, creating an equal superposition over all basis states:\n",
    "\n",
    "$$\n",
    "H^{\\otimes n} |0\\rangle = \\frac{1}{2^{n/2}}\\sum_{y=0}^{2^n-1}|y\\rangle.\n",
    "$$\n",
    "\n",
    "Now we need to insert the correct phase coefficients. This is done by a sequence of Controlled-$U^k$ operations, where the qubits of $y$ are the controls and the $U^k$ operations act on $|\\psi \\rangle$.\n",
    "\n",
    "Let's try to implement this part of the procedure in Cirq, and then put it together with the inverse QFT from above. For the gate $U$ we'll pick the single-qubit operation\n",
    "\n",
    "$$\n",
    "U = Z^{2\\theta} = \\begin{bmatrix}\n",
    "1 & 0 \\\\\n",
    "0 & e^{2\\pi i \\theta }\n",
    "\\end{bmatrix}\n",
    "$$\n",
    "\n",
    "for $\\theta \\in [0,1)$. This is just for simplicity and ease of testing. You are invited to write an implementation that accepts an arbitrary $U$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.412151Z",
     "iopub.status.busy": "2023-09-19T09:08:17.411897Z",
     "iopub.status.idle": "2023-09-19T09:08:17.415298Z",
     "shell.execute_reply": "2023-09-19T09:08:17.414716Z"
    },
    "id": "856ededbc425"
   },
   "outputs": [],
   "source": [
    "\"\"\"Set up the unitary and number of bits to use in phase estimation.\"\"\"\n",
    "# Value of θ which appears in the definition of the unitary U above.\n",
    "# Try different values.\n",
    "theta = 0.234\n",
    "\n",
    "# Define the unitary U.\n",
    "U = cirq.Z ** (2 * theta)\n",
    "\n",
    "# Accuracy of the estimate for theta. Try different values.\n",
    "n_bits = 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fbf130991336"
   },
   "source": [
    "Now we can build the first part of the circuit (up until the inverse QFT) for phase estimation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.418404Z",
     "iopub.status.busy": "2023-09-19T09:08:17.417897Z",
     "iopub.status.idle": "2023-09-19T09:08:17.424550Z",
     "shell.execute_reply": "2023-09-19T09:08:17.423945Z"
    },
    "id": "OIN8QfUeJyI9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0: ───H───@──────────────────────────────\n",
      "          │\n",
      "1: ───H───┼──────────@───────────────────\n",
      "          │          │\n",
      "2: ───H───┼──────────┼─────────@─────────\n",
      "          │          │         │\n",
      "u: ───────Z^-0.128───Z^0.936───Z^0.468───\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Build the first part of the circuit for phase estimation.\"\"\"\n",
    "# Get qubits for the phase estimation circuit.\n",
    "qubits = cirq.LineQubit.range(n_bits)\n",
    "u_bit = cirq.NamedQubit('u')\n",
    "\n",
    "# Build the first part of the phase estimation circuit.\n",
    "phase_estimator = cirq.Circuit(cirq.H.on_each(*qubits))\n",
    "\n",
    "for i, bit in enumerate(qubits):\n",
    "    phase_estimator.append(cirq.ControlledGate(U).on(bit, u_bit) ** (2 ** (n_bits - i - 1)))\n",
    "\n",
    "print(phase_estimator)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QTVJnv8Yx5bm"
   },
   "source": [
    "The next step is to perform the inverse QFT on the estimation qubits and measure them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.427619Z",
     "iopub.status.busy": "2023-09-19T09:08:17.427093Z",
     "iopub.status.idle": "2023-09-19T09:08:17.434990Z",
     "shell.execute_reply": "2023-09-19T09:08:17.434344Z"
    },
    "id": "8KCn-gjxM2H9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                         ┌────────┐\n",
      "0: ───H───@──────────H─────────@──────────@────────────────────────M('m')───\n",
      "          │                    │          │                        │\n",
      "1: ───H───┼──────────@─────────@^-0.5─────┼──────H────@────────────M────────\n",
      "          │          │                    │           │            │\n",
      "2: ───H───┼──────────┼─────────@──────────@^-0.25─────@^-0.5───H───M────────\n",
      "          │          │         │\n",
      "u: ───────Z^-0.128───Z^0.936───Z^0.468──────────────────────────────────────\n",
      "                                         └────────┘\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Build the last part of the circuit (inverse QFT) for phase estimation.\"\"\"\n",
    "# Do the inverse QFT.\n",
    "phase_estimator.append(make_qft_inverse(qubits[::-1]))\n",
    "\n",
    "# Add measurements to the end of the circuit\n",
    "phase_estimator.append(cirq.measure(*qubits, key='m'))\n",
    "print(phase_estimator)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "smlXIG1QyPyR"
   },
   "source": [
    "The initial state for `u_bit` is the $|0\\rangle$ state, but the phase for this state is trivial with the operator we chose. Inserting a Pauli $X$ operator at the beginning of the circuit changes this to the $|1\\rangle$ state, which has the nontrivial $\\theta$ phase. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.438166Z",
     "iopub.status.busy": "2023-09-19T09:08:17.437591Z",
     "iopub.status.idle": "2023-09-19T09:08:17.444839Z",
     "shell.execute_reply": "2023-09-19T09:08:17.444244Z"
    },
    "id": "g_rNMrkXPJ0R"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                         ┌────────┐\n",
      "0: ───H───@──────────H─────────@──────────@────────────────────────M('m')───\n",
      "          │                    │          │                        │\n",
      "1: ───H───┼──────────@─────────@^-0.5─────┼──────H────@────────────M────────\n",
      "          │          │                    │           │            │\n",
      "2: ───H───┼──────────┼─────────@──────────@^-0.25─────@^-0.5───H───M────────\n",
      "          │          │         │\n",
      "u: ───X───Z^-0.128───Z^0.936───Z^0.468──────────────────────────────────────\n",
      "                                         └────────┘\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Set the input state of the eigenvalue register.\"\"\"\n",
    "# Add gate to change initial state to |1>.\n",
    "phase_estimator.insert(0, cirq.X(u_bit))\n",
    "\n",
    "print(phase_estimator)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "j2HIBKbEy7gV"
   },
   "source": [
    "Now we can instantiate a simulator and make measurements of the estimation qubits. Let the values of these measured qubits be $a_j \\in \\{0, 1\\}$. Then our $n$-bit approximation for $\\theta$ is given by\n",
    "\n",
    "$$\n",
    "\\theta \\approx \\sum_{j=0}^n a_j2^{-j}.\n",
    "$$\n",
    "\n",
    "We'll perform this conversion from bit values to $\\theta$-values and then print the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.447827Z",
     "iopub.status.busy": "2023-09-19T09:08:17.447441Z",
     "iopub.status.idle": "2023-09-19T09:08:17.456398Z",
     "shell.execute_reply": "2023-09-19T09:08:17.455830Z"
    },
    "id": "-pE7CC_uPfq2"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.25  0.25  0.25  0.25  0.25  0.625 0.25  0.25  0.25  0.25 ]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Simulate the circuit and convert from measured bit values to estimated θ values.\"\"\"\n",
    "# Simulate the circuit.\n",
    "sim = cirq.Simulator()\n",
    "result = sim.run(phase_estimator, repetitions=10)\n",
    "\n",
    "# Convert from output bitstrings to estimate θ values.\n",
    "theta_estimates = np.sum(2 ** np.arange(n_bits) * result.measurements['m'], axis=1) / 2**n_bits\n",
    "print(theta_estimates)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PMke93CrzezN"
   },
   "source": [
    "When `n_bits` is small, we don't get a very accurate estimate. To test the accuracy of the estimate vs. `n_bits`, let's pack all this up into a single function that lets us specify $\\theta$, the number of bits of accuracy we want in our approximation, and the number of repetitions of the algorithm to perform. For future purposes, let's also include an argument for the gate which acts on `u_bit` at the start of the circuit to prepare the eigenstate."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "42fa4bb40a20"
   },
   "source": [
    "### Exercise: Define a function for phase estimation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "945deff89431"
   },
   "source": [
    "You could just copy/paste from the previous cells, but it might be a useful exercise to write the whole thing from scratch without peeking."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.459703Z",
     "iopub.status.busy": "2023-09-19T09:08:17.459149Z",
     "iopub.status.idle": "2023-09-19T09:08:17.463521Z",
     "shell.execute_reply": "2023-09-19T09:08:17.462949Z"
    },
    "id": "t3EYxglfpgbh"
   },
   "outputs": [],
   "source": [
    "def phase_estimation(theta, n_bits, n_reps=10, prepare_eigenstate_gate=cirq.X):\n",
    "    \"\"\"Runs the phase estimate algorithm for unitary U=Z^{2θ} with n_bits qubits.\"\"\"\n",
    "    # Define qubit registers.\n",
    "    qubits = cirq.LineQubit.range(n_bits)\n",
    "    u_bit = cirq.NamedQubit('u')\n",
    "\n",
    "    # Define the unitary U.\n",
    "    U = cirq.Z ** (2 * theta)\n",
    "\n",
    "    # Your code here!\n",
    "    # ...\n",
    "    \n",
    "    # Gate to choose the initial state for the u_bit. Placing X here chooses the |1> state.\n",
    "    phase_estimator.insert(0, prepare_eigenstate_gate.on(u_bit))\n",
    "    \n",
    "    # You code here!\n",
    "    # theta_estimates = ...\n",
    "    \n",
    "    return theta_estimates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5cf9gaXCpmq4"
   },
   "source": [
    "#### Solution"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fC5oUxNtppVo"
   },
   "source": [
    "Here is a solution that just consists of what we did in previous cells all put together."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.466731Z",
     "iopub.status.busy": "2023-09-19T09:08:17.466169Z",
     "iopub.status.idle": "2023-09-19T09:08:17.472116Z",
     "shell.execute_reply": "2023-09-19T09:08:17.471485Z"
    },
    "id": "TXxJ_ZjeWFqy"
   },
   "outputs": [],
   "source": [
    "def phase_estimation(theta, n_bits, n_reps=10, prepare_eigenstate_gate=cirq.X):\n",
    "    # Define qubit registers.\n",
    "    qubits = cirq.LineQubit.range(n_bits)\n",
    "    u_bit = cirq.NamedQubit('u')\n",
    "\n",
    "    # Define the unitary U.\n",
    "    U = cirq.Z ** (2 * theta)\n",
    "\n",
    "    # Start with Hadamards on every qubit.\n",
    "    phase_estimator = cirq.Circuit(cirq.H.on_each(*qubits))\n",
    "\n",
    "    # Do the controlled powers of the unitary U.\n",
    "    for i, bit in enumerate(qubits):\n",
    "        phase_estimator.append(cirq.ControlledGate(U).on(bit, u_bit) ** (2 ** (n_bits - 1 - i)))\n",
    "        \n",
    "    # Do the inverse QFT.\n",
    "    phase_estimator.append(make_qft_inverse(qubits[::-1]))\n",
    "\n",
    "    # Add measurements.\n",
    "    phase_estimator.append(cirq.measure(*qubits, key='m'))\n",
    "\n",
    "    # Gate to choose the initial state for the u_bit. Placing X here chooses the |1> state.\n",
    "    phase_estimator.insert(0, prepare_eigenstate_gate.on(u_bit))\n",
    "\n",
    "    # Code to simulate measurements\n",
    "    sim = cirq.Simulator()\n",
    "    result = sim.run(phase_estimator, repetitions=n_reps)\n",
    "\n",
    "    # Convert measurements into estimates of theta\n",
    "    theta_estimates = np.sum(2**np.arange(n_bits)*result.measurements['m'], axis=1)/2**n_bits\n",
    "\n",
    "    return theta_estimates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dbb852dc515a"
   },
   "source": [
    "Now we can easily examine the accuracy of the estimate vs `n_bits`. We do so for a variety of values for `n_bits` in the following cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.475146Z",
     "iopub.status.busy": "2023-09-19T09:08:17.474619Z",
     "iopub.status.idle": "2023-09-19T09:08:17.665761Z",
     "shell.execute_reply": "2023-09-19T09:08:17.665078Z"
    },
    "id": "5749cf9469da"
   },
   "outputs": [],
   "source": [
    "\"\"\"Analyze convergence vs n_bits.\"\"\"\n",
    "# Set the value of theta. Try different values.\n",
    "theta = 0.123456\n",
    "\n",
    "max_nvals = 16\n",
    "nvals = np.arange(1, max_nvals, step=1)\n",
    "\n",
    "# Get the estimates at each value of n.\n",
    "estimates = []\n",
    "for n in nvals:\n",
    "    estimate = phase_estimation(theta=theta, n_bits=n, n_reps=1)[0]\n",
    "    estimates.append(estimate)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "e3fc074b1e6c"
   },
   "source": [
    "And now we make a plot of the $\\theta$ estimates vs. the number of bits."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:17.669055Z",
     "iopub.status.busy": "2023-09-19T09:08:17.668819Z",
     "iopub.status.idle": "2023-09-19T09:08:18.065685Z",
     "shell.execute_reply": "2023-09-19T09:08:18.064859Z"
    },
    "id": "a8abf4de37bf"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\"\"\"Plot the results.\"\"\"\n",
    "plt.style.use(\"seaborn-v0_8-whitegrid\")\n",
    "\n",
    "plt.plot(nvals, estimates, \"--o\", label=\"Phase estimation\")\n",
    "plt.axhline(theta, label=\"True value\", color=\"black\")\n",
    "\n",
    "plt.legend()\n",
    "plt.xlabel(\"Number of bits\")\n",
    "plt.ylabel(r\"$\\theta$\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4e1e728d41d8"
   },
   "source": [
    "#### Phase Estimation Without an Eigenstate"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cr-NVLG2loo7"
   },
   "source": [
    "What if the input to the circuit was not an eigenstate of $U$ at all? We can always decompose such a state in the eigenbasis of $U$ as\n",
    "\n",
    "$$\n",
    "\\sum_j \\alpha_j|u_j\\rangle\n",
    "$$\n",
    "\n",
    "where $U |u_j\\rangle = e^{2 \\pi i \\theta_j} |u_j\\rangle$. Then each time we run the circuit we will get an $n$-bit estimate of one of the $\\theta_j$ chosen at random, and the probability of choosing a particular $\\theta_j$ is given by $|\\alpha_j|^2$. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8f227e1689eb"
   },
   "source": [
    "One simple test of this is to modify our above code to pass the state\n",
    "\n",
    "$$\n",
    "|+\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}\n",
    "$$\n",
    "\n",
    "into the phase estimator for $Z^{2\\theta}$. The state $|0\\rangle$ has eigenvalue $1$ while the state $|1\\rangle$ has eigenvalue $e^{2\\pi i \\theta_j}$. We can do this with the `prepare_eigenstate_gate` argument to the `phase_estimation_function`, as shown below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.069312Z",
     "iopub.status.busy": "2023-09-19T09:08:18.069021Z",
     "iopub.status.idle": "2023-09-19T09:08:18.079287Z",
     "shell.execute_reply": "2023-09-19T09:08:18.078677Z"
    },
    "id": "3790a1ba19ac"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.125 0.125 0.    0.    0.    0.    0.125 0.    0.125 0.125]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Run phase estimation without starting in an eigenstate.\"\"\"\n",
    "# Value of theta.\n",
    "theta = 0.123456\n",
    "\n",
    "# Number of qubits.\n",
    "n = 4\n",
    "\n",
    "# Run phase estimation starting in the state H|0⟩ = |+⟩.\n",
    "res = phase_estimation(theta=theta, n_bits=n, n_reps=10, prepare_eigenstate_gate=cirq.H)\n",
    "print(res)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wzTnTpC2oL79"
   },
   "source": [
    "Notice that roughly half of the measurements yielded the estimate $0$ (which corresponds to the eigenvalue $1$) and roughly half yield the estimate of `theta`. This is expected because the initial state $|+\\rangle$ is an equal superposition of the two eigenstates of $U = Z^{2 \\theta}$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9dd855b44451"
   },
   "source": [
    "Often we won't be able to prepare an exact eigenstate of the operator $U$ we are interested in, so it's very useful to know about this feature of phase estimation. This is crucial for understanding [Shor's algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm), for instance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zQzAYK-VNVDu"
   },
   "source": [
    "### Exercise: Quantum Fourier transform with unreversed output\n",
    "\n",
    "As mentioned, the `cirq.qft` function has an argument `without_reverse` for whether or not to reverse the output bits. Add a similar argument to our `make_qft` function which does the same thing. You may want to consider using SWAP gates."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EG6cDFWJk-ZS"
   },
   "source": [
    "### Exercise: Phase estimation with arbitrary $U$\n",
    "\n",
    "Try to implement the phase estimation algorithm in a way that an arbitrary gate $U$ can be supplied and tested. After you've done that, you can test the algorithm on some of your favorite two- or three-qubit gates."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9WKZoaav0dtt"
   },
   "source": [
    "### Exercise: QFT and phase estimation with adjacency constraints\n",
    "\n",
    "Often on a real machine we can't execute two-qubit gates between qubits that are not right next to each other. You'll have noticed that the circuits we defined above involves connections between many different pairs of qubits, which will likely not all be near each other when we try to run the circuit on an actual chip. See if you can modify the examples we went through above in such a way that Cirq validates them for use on the Sycamore chip."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fc3c5c22fb96"
   },
   "source": [
    "## Grover's algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ksA-fvrZaT5g"
   },
   "source": [
    "Consider bitstrings of length $n$ and let $x' \\in \\{0, 1\\}^{n}$ be a \"marked\" bitstring we wish to find. Grover's algorithm takes a black-box oracle implementing a function $f : \\{0, 1\\}^n \\rightarrow \\{0, 1\\}$ defined by \n",
    "\n",
    "$$\n",
    "f(x) = 1\\text{ if } x = x',~~~~ f(x) = 0 \\text{ if } x \\neq x'\n",
    "$$ \n",
    "\n",
    "to find such a bitstring $x'$. Grover's algorithm uses $O(\\sqrt{N}$) operations and $O(N\\, \\log N$) gates and succeeds with probability $p \\geq 2/3$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "86599076105d"
   },
   "source": [
    "Below, we walk through a simple implementation of Grover's algorithm described in [this reference](https://arxiv.org/abs/1804.03719). This implementation only supports $n = 2$ (for which one application of the Grover iteration is enough).\n",
    "\n",
    "First we define our qubit registers. We use $n = 2$ bits in one register and an additional ancilla qubit for phase kickback."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.083366Z",
     "iopub.status.busy": "2023-09-19T09:08:18.082854Z",
     "iopub.status.idle": "2023-09-19T09:08:18.086351Z",
     "shell.execute_reply": "2023-09-19T09:08:18.085710Z"
    },
    "id": "dae0e3e0d1bf"
   },
   "outputs": [],
   "source": [
    "\"\"\"Get qubits to use in the circuit for Grover's algorithm.\"\"\"\n",
    "# Number of qubits n.\n",
    "nqubits = 2\n",
    "\n",
    "# Get qubit registers.\n",
    "qubits = cirq.LineQubit.range(nqubits)\n",
    "ancilla = cirq.NamedQubit(\"Ancilla\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4a55fe70ea9a"
   },
   "source": [
    "We now define a generator to yield the operations for the oracle. As discussed in the above reference, the oracle can be implemented by a Toffoli gate if all the bits in $x'$ are $1$. If some bits are $0$, we do an \"open control\" (control on the $|0\\rangle$ state) for these bits. This can be accomplished by flipping every $0$ bit with $X$ gates, performing a Tofolli, then undoing the $X$ gates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.089692Z",
     "iopub.status.busy": "2023-09-19T09:08:18.089139Z",
     "iopub.status.idle": "2023-09-19T09:08:18.093537Z",
     "shell.execute_reply": "2023-09-19T09:08:18.092937Z"
    },
    "id": "0425db9fa9b0"
   },
   "outputs": [],
   "source": [
    "def make_oracle(qubits, ancilla, xprime):\n",
    "    \"\"\"Implements the function {f(x) = 1 if x == x', f(x) = 0 if x != x'}.\"\"\"\n",
    "    # For x' = (1, 1), the oracle is just a Toffoli gate.\n",
    "    # For a general x', we negate the zero bits and implement a Toffoli.\n",
    "    \n",
    "    # Negate zero bits, if necessary.\n",
    "    yield (cirq.X(q) for (q, bit) in zip(qubits, xprime) if not bit)\n",
    "    \n",
    "    # Do the Toffoli.\n",
    "    yield (cirq.TOFFOLI(qubits[0], qubits[1], ancilla))\n",
    "    \n",
    "    # Negate zero bits, if necessary.\n",
    "    yield (cirq.X(q) for (q, bit) in zip(qubits, xprime) if not bit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "e345344581e3"
   },
   "source": [
    "Now that we have a function to implement the oracle, we can construct a function to implement one round of Grover's iteration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.096810Z",
     "iopub.status.busy": "2023-09-19T09:08:18.096286Z",
     "iopub.status.idle": "2023-09-19T09:08:18.101946Z",
     "shell.execute_reply": "2023-09-19T09:08:18.101307Z"
    },
    "id": "6a8deed363e0"
   },
   "outputs": [],
   "source": [
    "def grover_iteration(qubits, ancilla, oracle):\n",
    "    \"\"\"Performs one round of the Grover iteration.\"\"\"\n",
    "    circuit = cirq.Circuit()\n",
    "\n",
    "    # Create an equal superposition over input qubits.\n",
    "    circuit.append(cirq.H.on_each(*qubits))\n",
    "    \n",
    "    # Put the output qubit in the |-⟩ state.\n",
    "    circuit.append([cirq.X(ancilla), cirq.H(ancilla)])\n",
    "\n",
    "    # Query the oracle.\n",
    "    circuit.append(oracle)\n",
    "\n",
    "    # Construct Grover operator.\n",
    "    circuit.append(cirq.H.on_each(*qubits))\n",
    "    circuit.append(cirq.X.on_each(*qubits))\n",
    "    circuit.append(cirq.H.on(qubits[1]))\n",
    "    circuit.append(cirq.CNOT(qubits[0], qubits[1]))\n",
    "    circuit.append(cirq.H.on(qubits[1]))\n",
    "    circuit.append(cirq.X.on_each(*qubits))\n",
    "    circuit.append(cirq.H.on_each(*qubits))\n",
    "\n",
    "    # Measure the input register.\n",
    "    circuit.append(cirq.measure(*qubits, key=\"result\"))\n",
    "\n",
    "    return circuit"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "992a9b48b6af"
   },
   "source": [
    "We now select the bitstring $x'$ at random."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.105234Z",
     "iopub.status.busy": "2023-09-19T09:08:18.104604Z",
     "iopub.status.idle": "2023-09-19T09:08:18.108686Z",
     "shell.execute_reply": "2023-09-19T09:08:18.108065Z"
    },
    "id": "033b376b387d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Marked bitstring: [0, 0]\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Select a 'marked' bitstring x' at random.\"\"\"\n",
    "xprime = [random.randint(0, 1) for _ in range(nqubits)]\n",
    "print(f\"Marked bitstring: {xprime}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6967a1252d03"
   },
   "source": [
    "And now create the circuit for Grover's algorithm."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.112139Z",
     "iopub.status.busy": "2023-09-19T09:08:18.111538Z",
     "iopub.status.idle": "2023-09-19T09:08:18.119240Z",
     "shell.execute_reply": "2023-09-19T09:08:18.118562Z"
    },
    "id": "gaUMhMV0aaVB"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Circuit for Grover's algorithm:\n",
      "0: ─────────H───X───@───X───H───X───────@───X───H───────M('result')───\n",
      "                    │                   │               │\n",
      "1: ─────────H───X───@───X───H───X───H───X───H───X───H───M─────────────\n",
      "                    │\n",
      "Ancilla: ───X───H───X─────────────────────────────────────────────────\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Create the circuit for Grover's algorithm.\"\"\"\n",
    "# Make oracle (black box)\n",
    "oracle = make_oracle(qubits, ancilla, xprime)\n",
    "\n",
    "# Embed the oracle into a quantum circuit implementing Grover's algorithm.\n",
    "circuit = grover_iteration(qubits, ancilla, oracle)\n",
    "print(\"Circuit for Grover's algorithm:\")\n",
    "print(circuit)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bb73c2e93119"
   },
   "source": [
    "All that is left is to simulate the circuit and check if the sampled bitstring(s) match with the marked bitstring $x'$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-19T09:08:18.122253Z",
     "iopub.status.busy": "2023-09-19T09:08:18.122002Z",
     "iopub.status.idle": "2023-09-19T09:08:18.131692Z",
     "shell.execute_reply": "2023-09-19T09:08:18.131048Z"
    },
    "id": "18c803b9ca8b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sampled results:\n",
      "Counter({'00': 10})\n",
      "\n",
      "Most common bitstring: 00\n",
      "Found a match? True\n"
     ]
    }
   ],
   "source": [
    "\"\"\"Simulate the circuit for Grover's algorithm and check the output.\"\"\"\n",
    "# Helper function.\n",
    "def bitstring(bits):\n",
    "    return \"\".join(str(int(b)) for b in bits)\n",
    "\n",
    "# Sample from the circuit a couple times.\n",
    "simulator = cirq.Simulator()\n",
    "result = simulator.run(circuit, repetitions=10)\n",
    "\n",
    "# Look at the sampled bitstrings.\n",
    "frequencies = result.histogram(key=\"result\", fold_func=bitstring)\n",
    "print('Sampled results:\\n{}'.format(frequencies))\n",
    "\n",
    "# Check if we actually found the secret value.\n",
    "most_common_bitstring = frequencies.most_common(1)[0][0]\n",
    "print(\"\\nMost common bitstring: {}\".format(most_common_bitstring))\n",
    "print(\"Found a match? {}\".format(most_common_bitstring == bitstring(xprime)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8e29ca81ae30"
   },
   "source": [
    "We see that we indeed found the marked bitstring $x'$. One can rerun these cells to select a new bitstring $x'$ and check that Grover's algorithm can again find it."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [
    "GbFgwEIW83qL",
    "vum44toJd_bb",
    "QbpYdr9Ngoyq",
    "96A0m3YUZ8D9",
    "5cf9gaXCpmq4",
    "pnddp79QmZAl"
   ],
   "name": "textbook_algorithms.ipynb",
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
