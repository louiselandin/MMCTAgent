
class NLIReward:
#     bad implementation of NLI reward
#     batch_of_pairs = [
#     {'text':"""You have to follow the following step and give a good response on what is the observation Step 1: Determine if the bottles in each image have the same design or branding.
# Query (left): Are all the bottles in this image from the same brand, based on their design and labeling?
# Query (right): Are all the bottles in this image from the same brand, based on their design and labeling?""", 'text_pair':"""Thought: I need to use a tool to help me answer the question.
# Action: vit
# Action Input: {'query': 'Are all the bottles in this image from the same brand, based on their design and labeling?', 'selected_image': 'left'}
# [0m[1;3;34mObservation: Yes, all the bottles in the image are from the same brand, based on their design and labeling."""},
#     {'text':"""You have to follow the following step and give a good response on what is the observation  2. Determine the brand of the bottles in the right image.
#    Query: "In the right image, what brand are the bottles?" """, 'text_pair':"""Thought: (Implicit) I can answer without any more tools!
# Response:  In the right image, what brand are the bottles?"""},
#     {'text':'potatoes are awesome.','text_pair': 'I like to run.'},
#     {'text':'Mars is very far from earth.', 'text_pair':'Mars is very close.'},
# ]
#     prompt = """
#         A agent is given an instruction and it can respond in terms of Action or a Thought, your task is to classify it into one of the Three options
#          1) "Entails" : If the following response is the suitable response for the instruction
#          2) "Contradiction": If the following response is not following the response properly
#          3) "Neutral": If the response is still trying to think through the instruction
         
#          This is the following instruction given: {}
#          The response of the Agent is: {}
#          Please only answer in one word which should be one of the 3 options given above"""
#         questions = []
#         for pairs in batch_of_pairs:
#             questions.append(prompt.format(pairs["text"], pairs["text_pair"]))
#         print(questions)
#             from transformers import AutoModelForCausalLM, AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
#         model = AutoModelForCausalLM.from_pretrained(
#         "stabilityai/stablelm-3b-4e1t",
#         trust_remote_code=True,
#         torch_dtype="auto",
#         )
#         model.cuda()
#         for question in questions:
#             inputs = tokenizer(question, return_tensors="pt").to("cuda")
#             tokens = model.generate(
#             **inputs,
#             max_new_tokens=64,
#             temperature=0.75,
#             top_p=0.95,
#             do_sample=True,
#             )
#             print(tokenizer.decode(tokens[0], skip_special_tokens=True))
    def __init__(self, *args, **kwargs ):
        raise NotImplementedError("This is not implemented yet because results are bad")