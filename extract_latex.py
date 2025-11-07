# %%
import re, pyperclip

def c(text: str):
    # 1. Replace escaped backslashes and symbols
    text = (text
        .replace(r'\textbackslash{}', '\\')
        .replace(r'\_', '_')
        .replace(r'\^', '^')
        .replace(r'\{', '{')
        .replace(r'\}', '}')
    )
    # 2. Remove duplicate unicode + LaTeX repetitions (like θ0\theta_0θ0)
    # text = re.sub(r'([A-Za-zα-ωΑ-Ωθ-ϕ\d])\\theta_0\1', r'\\theta_0', text)
    # text = re.sub(r'([A-Za-zα-ωΑ-Ωθ-ϕ\d])\\theta_1\1', r'\\theta_1', text)

    # General pattern: repeated unicode–LaTeX–unicode triplets
    # text = re.sub(r'([^\s])\\([a-zA-Z_]+)[^\s]', r'\\\2', text)

    # # 3. Trim and normalize spaces
    # text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    
    # 2. Print actual LaTeX string
    print(text)
    pyperclip.copy(text)


    # 3. Copy the *printed version* (the same that appears in terminal)
    # pyperclip.copy(text.encode('utf-8').decode('unicode_escape'))


# %%
import re

# s = r"\textbackslash{}frac\{\partial K\}{\textbackslash{}partial t\} = K\_{\textbackslash{}theta\_0\}\, \textbackslash{}nu\_0 (e\^{-\textbackslash{}theta\_0 + \textbackslash{}theta\_1\} - 1) + K\_{\textbackslash{}theta\_1\}\, \textbackslash{}nu\_1 (e\^{\textbackslash{}theta\_0 - \textbackslash{}theta\_1\} - 1)"
s = r"\textbackslash{}kappa\_{1,0\}(t) = K\_{\textbackslash{}theta\_0\}(0,0,t) = \textbackslash{}mathbb\{E\}[M(t)], \textbackslash{}qquad \textbackslash{}kappa\_{1,1\}(t) = K\_{\textbackslash{}theta\_1\}(0,0,t) = \textbackslash{}mathbb\{E\}[U(t)]"

s=r'Differentiate both sides of the PDE w.r.t. θ0\textbackslash{}theta\_0θ0 and evaluate at θ0=θ1=0\textbackslash{}theta\_0=\textbackslash{}theta\_1=0θ0=θ1=0:'
print()
# restore all escaped symbols
s = (
    s.replace(r'\textbackslash{}', '\\')
         .replace(r'\_', '_')
         .replace(r'\^', '^')
         .replace(r'\{', '{')
         .replace(r'\}', '}')
)

print(s)
