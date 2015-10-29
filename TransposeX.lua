local TransposeX, parent = torch.class('nn.TransposeX', 'nn.Module')

function TransposeX:__init()
   parent.__init(self)
end

function TransposeX:updateOutput(input)
  assert(input:dim() == 3, 'dim must be 3')
  local channel = input:size()[1]
  local height  = input:size()[2]
  local width   = input:size()[3]
  assert( height == 1, 'height must be 1')
  self.output = input:resize(channel,width):t()
  return self.output
end

function TransposeX:updateGradInput(input, gradOutput)
   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end
